"""Graph Storage Module.

Stores extracted entities and relationships in Neo4j with provenance tracking.
"""

import asyncio
import logging
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from backend.errors import Neo4jQueryError
from backend.interfaces import EmbeddingProvider, GraphConnection

if TYPE_CHECKING:
    from backend.knowledge.curation.bucket1_reader import ParsedIdentity, ParsedUser

logger = logging.getLogger(__name__)

# Default ontology version -- keep in sync with KnowledgeConfig.ontology_version.
# GraphStore does not import KnowledgeConfig to avoid circular dependencies;
# the factory injects the live value via the `ontology_version` constructor param.
_DEFAULT_ONTOLOGY_VERSION = "1.1.0"

# ADR-009 v1.1: user-facing relationship types allowed during graph-hop expansion.
# Mirrors the full user-facing edge set in backend/knowledge/ontologies/v1_0_0.py.
# Provenance edge types (DERIVED_FROM, EXTRACTED_FROM, LEARNED_FROM, ABOUT,
# SOURCED_FROM, REFERENCES) are intentionally excluded to preserve the
# :__Entity__ vs :__Provenance__ separation at hop 2+.
# SUPERSEDES is also excluded — it marks knowledge as out-of-date; surfacing
# superseded relationships via retrieval would mislead. Out-of-date facts
# should be filtered upstream by status, not traversed.
#
# Cluster 1 additions: IMPLEMENTED_WITH / MIST_HAS_TRAIT / MIST_HAS_CAPABILITY /
# MIST_HAS_PREFERENCE link MistIdentity to external-domain targets (Concept,
# Topic, Skill, Preference, Technology) grown via extraction. They must be
# in the allowlist so seed-expansion anchored at mist-identity can reach
# extracted self-model facts at hop >=1.
_USER_FACING_REL_TYPES: list[str] = [
    # MIST self-model edges (MistIdentity -> internal MistTrait/MistCapability/
    # MistPreference/MistUncertainty targets; seeded at startup).
    "HAS_TRAIT",
    "HAS_CAPABILITY",
    "HAS_PREFERENCE",
    "IS_UNCERTAIN_ABOUT",
    "ADAPTED_FOR",
    "LEARNED_SELF",
    # MIST-scope external edges (Cluster 1). MistIdentity -> external-domain
    # Concept / Topic / Skill / Preference / Technology targets grown via
    # extraction. Included so seed-expansion from mist-identity reaches
    # extracted facts like "MIST uses LanceDB" or "MIST is curious".
    "IMPLEMENTED_WITH",
    "MIST_HAS_TRAIT",
    "MIST_HAS_CAPABILITY",
    "MIST_HAS_PREFERENCE",
    # User profile + activity edges
    "USES",
    "LEARNING",
    "WORKS_ON",
    "WORKS_AT",
    "WORKS_WITH",
    "KNOWS",
    "KNOWS_PERSON",
    "MEMBER_OF",
    "INTERESTED_IN",
    "HAS_GOAL",
    "PREFERS",
    "DISLIKES",
    "EXPERT_IN",
    "STRUGGLES_WITH",
    "DECIDED",
    "EXPERIENCED",
    # Structural / ontological edges between entities
    "IS_A",
    "PART_OF",
    "RELATED_TO",
    "DEPENDS_ON",
    "USED_FOR",
    # Post-MVP additive (2026-04-22): temporal + quantified + document edges.
    # Paired with the Date / Milestone / Metric / Document node additions.
    # Included in user-facing traversal because they carry semantic user
    # content (not provenance): "when did X happen", "what's X's benchmark
    # score", "what document is X about", "what came before X".
    "OCCURRED_ON",
    "HAS_METRIC",
    "REFERENCES_DOCUMENT",
    "PRECEDED_BY",
    # v1.1.0 additive (2026-05-06): mechanism / pattern / strategy / convention
    # predicates. Paired with the Pattern / Convention / Mechanism / Strategy /
    # DataStructure entity additions. All content-level edges between extracted
    # entities -- user-facing traversal so retrieval can pivot on "what
    # mechanism implements X", "what does Y operate on", "what improves Z",
    # etc. Provenance / ontology-metadata edges are gated separately.
    "MECHANISM_OF",
    "OPERATES_ON",
    "INPUT_TO",
    "IMPROVES",
    "COMPRISES",
    "APPLICABLE_TO",
    "STRATEGY_FOR",
    "NAMING_CONVENTION_OF",
]


class GraphStore:
    """Manages storage of knowledge graph in Neo4j.

    Handles:
    - Storing extracted entities and relationships
    - Provenance tracking (which conversation created which entity)
    - Versioning support (ontology versions)
    """

    def __init__(
        self,
        connection: GraphConnection,
        embedding_generator: EmbeddingProvider,
        ontology_version: str = _DEFAULT_ONTOLOGY_VERSION,
    ):
        """Initialize graph store with injected dependencies.

        Args:
            connection: Graph database connection (satisfies GraphConnection protocol).
            embedding_generator: Embedding provider (satisfies EmbeddingProvider protocol).
            ontology_version: Current ontology version string. Exposed via
                `current_ontology_version()` for GraphStoreProtocol callers.
                Defaults to the module constant which tracks KnowledgeConfig.
                The factory injects the live value from config so rebuilds
                always stamp the correct version on DERIVED_FROM edges.
        """
        self.connection = connection
        self.embedding_generator = embedding_generator
        self._vector_indexes_available: bool | None = None  # None = lazy-probe
        self._ontology_version: str = ontology_version

    @property
    def vector_indexes_available(self) -> bool:
        """Return True if the entity vector index exists in Neo4j.

        On first access, probes Neo4j's SHOW INDEXES to detect an existing
        vector index. Previously this flag was set only by
        `initialize_schema()`, which meant retrievers that built a fresh
        GraphStore via the factory always observed False even when the index
        was online. The lazy probe decouples "this instance ran init" from
        "the index exists in the database".
        """
        if self._vector_indexes_available is None:
            try:
                rows = self.connection.execute_query(
                    "SHOW INDEXES YIELD name, type, state "
                    "WHERE type = 'VECTOR' AND name = 'entity_embeddings'"
                )
                self._vector_indexes_available = bool(rows) and any(
                    r.get("state") == "ONLINE" for r in rows
                )
            except Neo4jQueryError:
                self._vector_indexes_available = False
        return self._vector_indexes_available

    def initialize_schema(self):
        """Create indexes and constraints in Neo4j.

        Sets up:
        - Uniqueness constraints on entity IDs
        - Vector indexes for semantic search
        - Indexes for fast lookups
        """
        logger.info("Initializing Neo4j schema...")

        self.connection.connect()

        # Create uniqueness constraints
        constraints = [
            "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:__Entity__) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT conversation_id_unique IF NOT EXISTS FOR (c:ConversationEvent) REQUIRE c.conversation_id IS UNIQUE",
            "CREATE CONSTRAINT utterance_id_unique IF NOT EXISTS FOR (u:Utterance) REQUIRE u.utterance_id IS UNIQUE",
            "CREATE CONSTRAINT source_id_unique IF NOT EXISTS FOR (s:SourceDocument) REQUIRE s.source_id IS UNIQUE",
            "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (c:DocumentChunk) REQUIRE c.chunk_id IS UNIQUE",
            "CREATE CONSTRAINT external_source_uri_unique IF NOT EXISTS FOR (es:ExternalSource) REQUIRE es.source_uri IS UNIQUE",
            "CREATE CONSTRAINT vector_chunk_store_id_unique IF NOT EXISTS FOR (vc:VectorChunk) REQUIRE vc.vector_store_id IS UNIQUE",
            "CREATE CONSTRAINT provenance_id_unique IF NOT EXISTS FOR (p:__Provenance__) REQUIRE p.id IS UNIQUE",
        ]

        for constraint in constraints:
            try:
                self.connection.execute_write(constraint)
                logger.debug(f"Created constraint: {constraint[:50]}...")
            except Neo4jQueryError as e:
                logger.warning(f"Constraint may already exist: {e}")

        # Create indexes for performance
        indexes = [
            "CREATE INDEX entity_type_idx IF NOT EXISTS FOR (e:__Entity__) ON (e.entity_type)",
            "CREATE INDEX conversation_timestamp_idx IF NOT EXISTS FOR (c:ConversationEvent) ON (c.timestamp)",
            "CREATE INDEX utterance_timestamp_idx IF NOT EXISTS FOR (u:Utterance) ON (u.timestamp)",
            "CREATE INDEX source_type_idx IF NOT EXISTS FOR (s:SourceDocument) ON (s.source_type)",
            "CREATE INDEX chunk_position_idx IF NOT EXISTS FOR (c:DocumentChunk) ON (c.position)",
            "CREATE INDEX source_hash_idx IF NOT EXISTS FOR (s:SourceDocument) ON (s.content_hash)",
            "CREATE INDEX external_source_type_idx IF NOT EXISTS FOR (es:ExternalSource) ON (es.source_type)",
            "CREATE INDEX vector_chunk_source_id_idx IF NOT EXISTS FOR (vc:VectorChunk) ON (vc.source_id)",
            "CREATE INDEX provenance_type_idx IF NOT EXISTS FOR (p:__Provenance__) ON (p.entity_type)",
        ]

        for index in indexes:
            try:
                self.connection.execute_write(index)
                logger.debug(f"Created index: {index[:50]}...")
            except Neo4jQueryError as e:
                logger.warning(f"Index may already exist: {e}")

        # Create vector indexes for semantic search
        # Note: Requires Neo4j 5.11+ with vector index support
        vector_indexes = [
            """
            CREATE VECTOR INDEX entity_embeddings IF NOT EXISTS
            FOR (e:__Entity__)
            ON e.embedding
            OPTIONS {
                indexConfig: {
                    `vector.dimensions`: 384,
                    `vector.similarity_function`: 'cosine'
                }
            }
            """,
            """
            CREATE VECTOR INDEX chunk_embeddings IF NOT EXISTS
            FOR (c:DocumentChunk)
            ON c.embedding
            OPTIONS {
                indexConfig: {
                    `vector.dimensions`: 384,
                    `vector.similarity_function`: 'cosine'
                }
            }
            """,
        ]

        successful_vector_indexes = 0
        for vector_index in vector_indexes:
            try:
                self.connection.execute_write(vector_index)
                logger.info("Created vector index")
                successful_vector_indexes += 1
            except Neo4jQueryError as e:
                logger.warning(f"Vector index creation failed (may not be supported): {e}")
                logger.warning("Vector search will not be available without Neo4j 5.11+")

        self._vector_indexes_available = successful_vector_indexes == len(vector_indexes)
        if not self._vector_indexes_available:
            logger.warning(
                "Not all vector indexes were created (%d/%d). "
                "Vector search methods will return empty results.",
                successful_vector_indexes,
                len(vector_indexes),
            )

        logger.info("Schema initialization complete")

    def store_conversation_event(
        self, conversation_id: str, user_id: str, timestamp: datetime | None = None
    ) -> str:
        """Store a conversation event.

        Args:
            conversation_id: Unique conversation identifier
            user_id: User identifier
            timestamp: Conversation timestamp

        Returns:
            Conversation event ID
        """
        if timestamp is None:
            timestamp = datetime.now()

        query = """
        MERGE (c:ConversationEvent {conversation_id: $conversation_id})
        ON CREATE SET
            c.user_id = $user_id,
            c.timestamp = datetime($timestamp),
            c.created_at = datetime()
        RETURN c.conversation_id AS id
        """

        params = {
            "conversation_id": conversation_id,
            "user_id": user_id,
            "timestamp": timestamp.isoformat(),
        }

        self.connection.execute_write(query, params)
        return conversation_id

    def store_utterance(
        self,
        utterance_id: str,
        conversation_id: str,
        text: str,
        timestamp: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Store a single utterance.

        Args:
            utterance_id: Unique utterance identifier
            conversation_id: Parent conversation ID
            text: Utterance text
            timestamp: Utterance timestamp
            metadata: Additional metadata

        Returns:
            Utterance ID
        """
        if timestamp is None:
            timestamp = datetime.now()

        # Build query conditionally based on metadata
        if metadata:
            import json

            # Convert metadata dict to JSON string (Neo4j doesn't support nested dicts)
            metadata_json = json.dumps(metadata)

            query = """
            MATCH (c:ConversationEvent {conversation_id: $conversation_id})
            MERGE (u:Utterance {utterance_id: $utterance_id})
            ON CREATE SET
                u.text = $text,
                u.timestamp = datetime($timestamp),
                u.metadata = $metadata,
                u.created_at = datetime()
            MERGE (u)-[:PART_OF]->(c)
            RETURN u.utterance_id AS id
            """
            params = {
                "utterance_id": utterance_id,
                "conversation_id": conversation_id,
                "text": text,
                "timestamp": timestamp.isoformat(),
                "metadata": metadata_json,
            }
        else:
            query = """
            MATCH (c:ConversationEvent {conversation_id: $conversation_id})
            MERGE (u:Utterance {utterance_id: $utterance_id})
            ON CREATE SET
                u.text = $text,
                u.timestamp = datetime($timestamp),
                u.created_at = datetime()
            MERGE (u)-[:PART_OF]->(c)
            RETURN u.utterance_id AS id
            """
            params = {
                "utterance_id": utterance_id,
                "conversation_id": conversation_id,
                "text": text,
                "timestamp": timestamp.isoformat(),
            }

        self.connection.execute_write(query, params)
        return utterance_id

    def store_source_document(
        self,
        source_id: str,
        file_path: str,
        source_type: str,
        content_hash: str,
        title: str | None = None,
        author: str | None = None,
        file_size: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Store a source document.

        .. deprecated::
            Use IngestionPipeline instead.

        Args:
            source_id: Unique source identifier
            file_path: Path or URI to source
            source_type: Type of source (markdown, pdf, web, upload)
            content_hash: SHA256 hash of content
            title: Document title
            author: Document author
            file_size: File size in bytes
            metadata: Additional metadata (will be JSON serialized)

        Returns:
            Source ID
        """
        import json
        import warnings

        warnings.warn(
            "GraphStore.store_source_document is deprecated. Use IngestionPipeline.",
            DeprecationWarning,
            stacklevel=2,
        )

        query = """
        MERGE (s:SourceDocument {source_id: $source_id})
        ON CREATE SET
            s.file_path = $file_path,
            s.source_type = $source_type,
            s.content_hash = $content_hash,
            s.title = $title,
            s.author = $author,
            s.file_size = $file_size,
            s.metadata = $metadata,
            s.ingested_at = datetime(),
            s.created_at = datetime()
        RETURN s.source_id AS id
        """

        params = {
            "source_id": source_id,
            "file_path": file_path,
            "source_type": source_type,
            "content_hash": content_hash,
            "title": title,
            "author": author,
            "file_size": file_size,
            "metadata": json.dumps(metadata) if metadata else None,
        }

        self.connection.execute_write(query, params)
        return source_id

    def store_document_chunk(
        self,
        chunk_id: str,
        source_id: str,
        text: str,
        position: int,
        embedding: list[float] | None = None,
        section_title: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Store a document chunk with optional embedding.

        .. deprecated::
            Use IngestionPipeline instead.

        Args:
            chunk_id: Unique chunk identifier
            source_id: Source document ID
            text: Chunk text
            position: Position in document
            embedding: Vector embedding for semantic search
            section_title: Section/header title
            metadata: Additional metadata (will be JSON serialized)

        Returns:
            Chunk ID
        """
        import json
        import warnings

        warnings.warn(
            "GraphStore.store_document_chunk is deprecated. Use IngestionPipeline.",
            DeprecationWarning,
            stacklevel=2,
        )

        # Calculate word and char counts
        word_count = len(text.split())
        char_count = len(text)

        # Generate embedding if not provided
        if embedding is None:
            embedding = self.embedding_generator.generate_embedding(text)

        query = """
        MATCH (s:SourceDocument {source_id: $source_id})
        MERGE (c:DocumentChunk {chunk_id: $chunk_id})
        ON CREATE SET
            c.text = $text,
            c.position = $position,
            c.word_count = $word_count,
            c.char_count = $char_count,
            c.section_title = $section_title,
            c.embedding = $embedding,
            c.metadata = $metadata,
            c.created_at = datetime()
        MERGE (c)-[:FROM_SOURCE]->(s)
        RETURN c.chunk_id AS id
        """

        params = {
            "chunk_id": chunk_id,
            "source_id": source_id,
            "text": text,
            "position": position,
            "word_count": word_count,
            "char_count": char_count,
            "section_title": section_title,
            "embedding": embedding,
            "metadata": json.dumps(metadata) if metadata else None,
        }

        self.connection.execute_write(query, params)
        return chunk_id

    def store_external_source(self, source_uri: str, source_type: str, **kwargs) -> str:
        """Store an external source provenance record.

        Creates or updates an ExternalSource node representing a document,
        MCP output, web resource, or other external data origin for the
        hybrid vector store architecture.

        Args:
            source_uri: Unique URI identifying the source.
            source_type: Type of source (document, mcp, web, etc.).
            **kwargs: Additional properties to store on the node.
                Only primitive types (str, int, float, bool) are accepted.

        Returns:
            The source_uri of the stored node.
        """
        # Filter kwargs to primitive types only
        safe_props = {k: v for k, v in kwargs.items() if isinstance(v, str | int | float | bool)}

        # Build dynamic property SET fragments
        prop_sets = "".join(f",\n            es.{k} = ${k}" for k in safe_props)

        query = f"""
        MERGE (es:ExternalSource {{source_uri: $source_uri}})
        ON CREATE SET
            es.source_type = $source_type,
            es.created_at = datetime(){prop_sets}
        ON MATCH SET
            es.source_type = $source_type,
            es.updated_at = datetime(){prop_sets}
        RETURN es.source_uri AS id
        """

        params = {"source_uri": source_uri, "source_type": source_type, **safe_props}

        self.connection.execute_write(query, params)
        return source_uri

    def store_vector_chunk_ref(self, vector_store_id: str, source_id: str, **kwargs) -> str:
        """Store a lightweight vector chunk reference node.

        Creates or updates a VectorChunk node that acts as a Neo4j-side
        pointer to a chunk stored in LanceDB. Unlike DocumentChunk, this
        node does NOT store text or embedding data.

        The `source_id` links to an ExternalSource node via its `source_uri`
        property.

        Args:
            vector_store_id: Unique ID matching the chunk in LanceDB.
            source_id: The `source_uri` of the parent ExternalSource.
            **kwargs: Additional properties to store on the node.
                Only primitive types (str, int, float, bool) are accepted.

        Returns:
            The vector_store_id of the stored node.
        """
        # Filter kwargs to primitive types only
        safe_props = {k: v for k, v in kwargs.items() if isinstance(v, str | int | float | bool)}

        # Build dynamic property SET fragments
        prop_sets = "".join(f",\n            vc.{k} = ${k}" for k in safe_props)

        query = f"""
        MERGE (vc:__Provenance__:VectorChunk {{vector_store_id: $vector_store_id}})
        ON CREATE SET
            vc.source_id = $source_id,
            vc.created_at = datetime(){prop_sets}
        ON MATCH SET
            vc.source_id = $source_id,
            vc.updated_at = datetime(){prop_sets}
        RETURN vc.vector_store_id AS id
        """

        params = {"vector_store_id": vector_store_id, "source_id": source_id, **safe_props}

        self.connection.execute_write(query, params)
        return vector_store_id

    def create_provenance_links(
        self,
        entity_ids: list[str],
        source_uri: str,
        source_type: str,
        chunk_ids: list[str] | None = None,
    ) -> None:
        """Link entities to their external source and optional vector chunks.

        Creates SOURCED_FROM edges from each entity to the ExternalSource node,
        and optionally REFERENCES edges from each entity to each VectorChunk.

        The entity-to-chunk linking uses a Cartesian product: every entity is
        linked to every chunk. This is intentional for document-level provenance.
        Per-entity chunk mapping is deferred to the curation pipeline (K-05).

        Args:
            entity_ids: List of `__Entity__` node IDs to link.
            source_uri: URI of the ExternalSource node.
            source_type: Type passed to `store_external_source` (document, mcp, web, etc.).
            chunk_ids: Optional list of VectorChunk `vector_store_id` values.
                An empty list is treated the same as None (no REFERENCES edges).
        """
        if not entity_ids:
            logger.debug("create_provenance_links called with empty entity_ids, skipping")
            return

        # Ensure ExternalSource node exists (MERGE, idempotent)
        self.store_external_source(source_uri, source_type)

        # Ensure VectorChunk nodes exist and are linked to the source
        if chunk_ids:
            for chunk_id in chunk_ids:
                self.store_vector_chunk_ref(chunk_id, source_uri)

        # Create SOURCED_FROM edges (batch via UNWIND)
        sourced_query = """
        UNWIND $entity_ids AS eid
        MATCH (e:__Entity__ {id: eid})
        MATCH (es:ExternalSource {source_uri: $source_uri})
        MERGE (e)-[r:SOURCED_FROM]->(es)
        ON CREATE SET r.created_at = datetime()
        ON MATCH SET r.updated_at = datetime()
        """
        self.connection.execute_write(
            sourced_query,
            {"entity_ids": entity_ids, "source_uri": source_uri},
        )

        # Create REFERENCES edges (Cartesian product -- intentional, document-level provenance)
        if chunk_ids:
            references_query = """
            UNWIND $entity_ids AS eid
            UNWIND $chunk_ids AS cid
            MATCH (e:__Entity__ {id: eid})
            MATCH (vc:VectorChunk {vector_store_id: cid})
            MERGE (e)-[r:REFERENCES]->(vc)
            ON CREATE SET r.created_at = datetime()
            ON MATCH SET r.updated_at = datetime()
            """
            self.connection.execute_write(
                references_query,
                {"entity_ids": entity_ids, "chunk_ids": chunk_ids},
            )

    def store_extracted_entities(
        self,
        graph_document,
        utterance_id: str | None = None,
        chunk_id: str | None = None,
        ontology_version: str = "1.0.0",  # Default to current ontology version
    ):
        """Store extracted entities and relationships from LLMGraphTransformer.

        Entities can be extracted from either:
        - Utterances (conversational knowledge)
        - DocumentChunks (document knowledge)

        Args:
            graph_document: GraphDocument from LLMGraphTransformer
            utterance_id: Source utterance ID (for conversational extraction)
            chunk_id: Source document chunk ID (for document extraction)
            ontology_version: Ontology version used for extraction

        Note: Must provide either utterance_id OR chunk_id, not both
        """
        if not utterance_id and not chunk_id:
            raise ValueError("Must provide either utterance_id or chunk_id")
        if utterance_id and chunk_id:
            raise ValueError("Cannot provide both utterance_id and chunk_id")

        source_id = utterance_id or chunk_id
        source_type = "utterance" if utterance_id else "chunk"

        logger.info(
            f"Storing {len(graph_document.nodes)} nodes and {len(graph_document.relationships)} relationships from {source_type}"
        )

        # Store nodes
        for node in graph_document.nodes:
            self._store_node(node, source_id, source_type, ontology_version)

        # Store relationships
        for relationship in graph_document.relationships:
            self._store_relationship(relationship, source_id, source_type, ontology_version)

    def store_validated_entities(
        self,
        entities: list[dict],
        relationships: list[dict],
        utterance_id: str,
        ontology_version: str = "1.0.0",
    ) -> None:
        """Store entities and relationships from ValidationResult format.

        Accepts the dict-based format from ExtractionPipeline's ValidationResult
        instead of GraphDocument with Node objects. Uses the same MERGE queries
        as store_extracted_entities.

        Args:
            entities: List of entity dicts with keys: id, type, name, confidence,
                source_type, aliases, description.
            relationships: List of relationship dicts with keys: source, target,
                type, confidence, source_type, temporal_status, context.
            utterance_id: Source utterance ID for provenance.
            ontology_version: Ontology version used for extraction.
        """
        logger.info(
            "Storing %d entities and %d relationships from validated extraction",
            len(entities),
            len(relationships),
        )

        for entity in entities:
            self._store_validated_node(entity, utterance_id, ontology_version)

        for rel in relationships:
            self._store_validated_relationship(rel, utterance_id, ontology_version)

    def _store_validated_node(self, entity: dict, utterance_id: str, ontology_version: str) -> None:
        """Store a single entity from dict format."""
        node_id = entity.get("id", "")
        entity_type = entity.get("type", "Unknown")
        display_name = entity.get("name", node_id)
        confidence = entity.get("confidence", 0.8)
        description = entity.get("description", "")

        embed_parts = [node_id]
        if entity_type and entity_type != "Unknown":
            embed_parts.append(entity_type)
        if description:
            embed_parts.append(description)
        embedding = self.embedding_generator.generate_embedding(" ".join(embed_parts))

        # The :User label is an invariant of the user node, not a seed-only
        # decoration: persona/identity reads anchor on it, and an
        # extraction-created label-less user node broke them in production
        # (deep review cypher-data-integrity-2a). SET is idempotent.
        user_label_set = "SET e:User" if node_id == "user" else ""
        query = f"""
        MATCH (u:Utterance {{utterance_id: $utterance_id}})
        MERGE (e:__Entity__ {{id: $node_id}})
        ON CREATE SET
            e.entity_type = $entity_type,
            e.display_name = $display_name,
            e.confidence = $confidence,
            e.ontology_version = $ontology_version,
            e.embedding = $embedding,
            e.description = $description,
            e.created_at = datetime()
        ON MATCH SET
            e.updated_at = datetime(),
            e.entity_type = CASE WHEN e.entity_type = 'Unknown'
                THEN $entity_type ELSE e.entity_type END,
            e.embedding = $embedding
        {user_label_set}
        MERGE (u)-[:HAS_ENTITY]->(e)
        """

        self.connection.execute_write(
            query,
            {
                "utterance_id": utterance_id,
                "node_id": node_id,
                "entity_type": entity_type,
                "display_name": display_name,
                "confidence": confidence,
                "ontology_version": ontology_version,
                "embedding": embedding,
                "description": description,
            },
        )

    def _store_validated_relationship(
        self, rel: dict, utterance_id: str, ontology_version: str
    ) -> None:
        """Store a single relationship from dict format."""
        source = rel.get("source", "")
        target = rel.get("target", "")
        rel_type = rel.get("type", "")
        confidence = rel.get("confidence", 0.8)

        sanitized_type = re.sub(r"[^A-Z_]", "", rel_type.upper())
        if not sanitized_type:
            logger.warning("Invalid relationship type '%s', skipping", rel_type)
            return

        query = f"""
        MATCH (s:__Entity__ {{id: $source}})
        MATCH (t:__Entity__ {{id: $target}})
        MERGE (s)-[r:{sanitized_type}]->(t)
        ON CREATE SET
            r.confidence = $confidence,
            r.ontology_version = $ontology_version,
            r.created_at = datetime()
        ON MATCH SET
            r.updated_at = datetime(),
            r.confidence = $confidence
        """

        self.connection.execute_write(
            query,
            {
                "source": source,
                "target": target,
                "confidence": confidence,
                "ontology_version": ontology_version,
            },
        )

    def _store_node(self, node, source_id: str, source_type: str, ontology_version: str):
        """Store a single entity node with provenance.

        Args:
            node: Node from GraphDocument
            source_id: Source ID (utterance_id or chunk_id)
            source_type: Type of source ("utterance" or "chunk")
            ontology_version: Ontology version
        """
        # Extract node properties
        node_id = node.id
        entity_type = node.type if hasattr(node, "type") else "Unknown"
        properties = getattr(node, "properties", {})

        # Generate embedding for semantic search
        # Build semantic text from available properties for richer embeddings
        embed_parts = [node_id]
        if entity_type and entity_type != "Unknown":
            embed_parts.append(entity_type)
        description = properties.get("description", "")
        if description:
            embed_parts.append(description)
        embedding_text = " ".join(embed_parts)
        embedding = self.embedding_generator.generate_embedding(embedding_text)

        # Build query with dynamic property setting
        # Neo4j doesn't allow nested dictionaries, so flatten properties
        property_sets = []
        params = {
            "source_id": source_id,
            "node_id": node_id,
            "entity_type": entity_type,
            "ontology_version": ontology_version,
            "embedding": embedding,
        }

        # Only allow safe property keys (alphanumeric + underscore, no Cypher special chars)
        SAFE_PROPERTY_KEY = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

        # Add each property individually if they exist
        if properties:
            for key, value in properties.items():
                if not SAFE_PROPERTY_KEY.match(key):
                    logger.warning("Unsafe property key rejected: %r", key)
                    continue
                # Only add primitive types (string, number, boolean)
                if isinstance(value, str | int | float | bool):
                    param_key = f"prop_{key}"
                    property_sets.append(f"e.{key} = ${param_key}")
                    params[param_key] = value

        # Build the ON CREATE SET clause
        base_sets = """
            e.entity_type = $entity_type,
            e.ontology_version = $ontology_version,
            e.embedding = $embedding,
            e.created_at = datetime()"""

        if property_sets:
            all_sets = base_sets + ",\n            " + ",\n            ".join(property_sets)
        else:
            all_sets = base_sets

        # Build the ON MATCH SET clause (update mutable properties on re-encounter)
        match_sets = """
            e.updated_at = datetime(),
            e.entity_type = CASE WHEN e.entity_type = 'Unknown' THEN $entity_type ELSE e.entity_type END,
            e.embedding = $embedding"""

        if property_sets:
            match_sets = match_sets + ",\n            " + ",\n            ".join(property_sets)

        # Different query based on source type
        if source_type == "utterance":
            query = f"""
            MATCH (u:Utterance {{utterance_id: $source_id}})
            MERGE (e:__Entity__ {{id: $node_id}})
            ON CREATE SET
                {all_sets}
            ON MATCH SET
                {match_sets}
            MERGE (u)-[:HAS_ENTITY]->(e)
            """
        else:  # chunk
            query = f"""
            MATCH (c:DocumentChunk {{chunk_id: $source_id}})
            MERGE (e:__Entity__ {{id: $node_id}})
            ON CREATE SET
                {all_sets}
            ON MATCH SET
                {match_sets}
            MERGE (e)-[:EXTRACTED_FROM]->(c)
            """

        self.connection.execute_write(query, params)
        logger.debug(f"Stored node: {node_id} ({entity_type}) from {source_type}")

    def _store_relationship(
        self, relationship, source_id: str, source_type: str, ontology_version: str
    ):
        """Store a single relationship between entities.

        Args:
            relationship: Relationship from GraphDocument
            source_id: Source ID (utterance_id or chunk_id)
            source_type: Type of source ("utterance" or "chunk")
            ontology_version: Ontology version
        """
        # Extract relationship details
        entity_source_id = relationship.source.id
        entity_target_id = relationship.target.id
        rel_type = relationship.type
        properties = getattr(relationship, "properties", {})

        # Sanitize relationship type for Cypher
        rel_type_safe = rel_type.replace(" ", "_").replace("-", "_").upper()

        # Validate against Cypher injection -- only alphanumeric + underscore allowed
        if not re.match(r"^[A-Z][A-Z0-9_]*$", rel_type_safe):
            logger.warning(f"Invalid relationship type rejected: {rel_type!r}")
            return

        # Build query with dynamic property setting
        property_sets = []
        params = {
            "source_id": entity_source_id,
            "target_id": entity_target_id,
            "ontology_version": ontology_version,
        }

        # Only allow safe property keys (alphanumeric + underscore, no Cypher special chars)
        SAFE_PROPERTY_KEY = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

        # Add each property individually if they exist
        if properties:
            for key, value in properties.items():
                if not SAFE_PROPERTY_KEY.match(key):
                    logger.warning("Unsafe property key rejected: %r", key)
                    continue
                # Only add primitive types (string, number, boolean)
                if isinstance(value, str | int | float | bool):
                    param_key = f"prop_{key}"
                    property_sets.append(f"r.{key} = ${param_key}")
                    params[param_key] = value

        # Build the ON CREATE SET clause
        base_sets = """
            r.ontology_version = $ontology_version,
            r.created_at = datetime()"""

        if property_sets:
            all_sets = base_sets + ",\n            " + ",\n            ".join(property_sets)
        else:
            all_sets = base_sets

        # Build the ON MATCH SET clause (update mutable properties on re-encounter)
        match_sets = """
            r.updated_at = datetime()"""

        if property_sets:
            match_sets = match_sets + ",\n            " + ",\n            ".join(property_sets)

        query = f"""
        MATCH (source:__Entity__ {{id: $source_id}})
        MATCH (target:__Entity__ {{id: $target_id}})
        MERGE (source)-[r:{rel_type_safe}]->(target)
        ON CREATE SET
            {all_sets}
        ON MATCH SET
            {match_sets}
        """

        self.connection.execute_write(query, params)
        logger.debug(f"Stored relationship: {entity_source_id} -[{rel_type}]-> {entity_target_id}")

    def get_entities_for_conversation(self, conversation_id: str) -> list[dict]:
        """Retrieve all entities extracted from a conversation.

        Args:
            conversation_id: Conversation identifier

        Returns:
            List of entity dictionaries
        """
        query = """
        MATCH (c:ConversationEvent {conversation_id: $conversation_id})
              <-[:PART_OF]-(u:Utterance)
              -[:HAS_ENTITY]->(e:__Entity__)
        RETURN DISTINCT
            e.id AS entity_id,
            e.entity_type AS entity_type,
            properties(e) AS properties,
            collect(DISTINCT u.utterance_id) AS source_utterances
        """

        params = {"conversation_id": conversation_id}
        results = self.connection.execute_query(query, params)

        return [dict(record) for record in results]

    def search_similar_entities(
        self, query_text: str, limit: int = 10, similarity_threshold: float = 0.7
    ) -> list[dict]:
        """Search for entities semantically similar to query text.

        Uses vector similarity search on entity embeddings.
        Requires Neo4j 5.11+ with vector index support.

        Args:
            query_text: Text to search for
            limit: Maximum number of results to return
            similarity_threshold: Minimum similarity score (0-1)

        Returns:
            List of SearchResult dictionaries with entity info and similarity scores

        Example:
            >>> results = graph_store.search_similar_entities("Python programming", limit=5)
            >>> for r in results:
            >>>     print(f"{r['entity_id']}: {r['similarity']:.3f}")
        """
        if not self.vector_indexes_available:
            logger.warning(
                "search_similar_entities called but vector indexes are not available; "
                "returning empty results"
            )
            return []

        # Generate embedding for query text
        query_embedding = self.embedding_generator.generate_embedding(query_text)

        # Vector similarity search using Neo4j's vector index
        # db.index.vector.queryNodes returns nodes sorted by similarity
        query = """
        CALL db.index.vector.queryNodes('entity_embeddings', $limit, $query_embedding)
        YIELD node, score
        WHERE score >= $similarity_threshold
        RETURN
            node.id AS entity_id,
            node.entity_type AS entity_type,
            score AS similarity,
            properties(node) AS properties
        ORDER BY score DESC
        """

        params = {
            "query_embedding": query_embedding,
            "limit": limit,
            "similarity_threshold": similarity_threshold,
        }

        try:
            results = self.connection.execute_query(query, params)
            return [dict(record) for record in results]
        except (Neo4jQueryError, Exception) as e:
            logger.warning(f"Vector search failed: {e}")
            logger.warning(
                "Disabling vector search. Ensure Neo4j 5.11+ is installed and "
                "vector indexes are created."
            )
            self._vector_indexes_available = False
            return []

    def search_document_chunks(
        self, query_text: str, limit: int = 5, similarity_threshold: float = 0.7
    ) -> list[dict]:
        """Search DocumentChunks using vector similarity (RAG retrieval).

        Uses vector similarity search on chunk embeddings to find
        relevant document passages.

        Args:
            query_text: Text to search for
            limit: Maximum number of chunks to return
            similarity_threshold: Minimum similarity score (0-1)

        Returns:
            List of dictionaries with chunk info and source metadata:
            [
                {
                    "chunk_id": "...",
                    "text": "...",
                    "similarity": 0.85,
                    "source_title": "...",
                    "source_file": "...",
                    "position": 0
                }
            ]
        """
        if not self.vector_indexes_available:
            logger.warning(
                "search_document_chunks called but vector indexes are not available; "
                "returning empty results"
            )
            return []

        # Generate embedding for query text
        query_embedding = self.embedding_generator.generate_embedding(query_text)

        # Vector similarity search on DocumentChunk embeddings
        query = """
        CALL db.index.vector.queryNodes('chunk_embeddings', $limit, $query_embedding)
        YIELD node, score
        WHERE score >= $similarity_threshold
        MATCH (s:SourceDocument)-[:FROM_SOURCE]-(node)
        RETURN
            node.chunk_id AS chunk_id,
            node.text AS text,
            node.position AS position,
            score AS similarity,
            s.title AS source_title,
            s.file_path AS source_file,
            s.source_type AS source_type
        ORDER BY score DESC
        """

        params = {
            "query_embedding": query_embedding,
            "limit": limit,
            "similarity_threshold": similarity_threshold,
        }

        try:
            results = self.connection.execute_query(query, params)
            return [dict(record) for record in results]
        except (Neo4jQueryError, Exception) as e:
            logger.warning(f"Document chunk search failed: {e}")
            logger.warning(
                "Disabling vector search. Ensure Neo4j 5.11+ is installed and "
                "chunk_embeddings vector index is created."
            )
            self._vector_indexes_available = False
            return []

    def get_entity_neighborhood(
        self, entity_id: str, max_hops: int = 2, relationship_types: list[str] | None = None
    ) -> list[dict]:
        """Get N-hop neighborhood around an entity.

        Returns all entities and relationships within N hops.

        Args:
            entity_id: Starting entity
            max_hops: Maximum traversal depth (1-3 recommended)
            relationship_types: Optional filter for specific relationship types

        Returns:
            List of dicts with structure:
            {
                'path_length': int,
                'source': str,
                'source_type': str,
                'relationship': str,
                'target': str,
                'target_type': str,
                'properties': dict
            }
        """
        if not isinstance(max_hops, int) or max_hops < 1 or max_hops > 5:
            raise ValueError(f"max_hops must be an integer between 1 and 5, got {max_hops}")

        # ADR-009 v1.1: always enforce user-facing rel-type allowlist.
        # Caller may pass a tighter subset; fall back to the module-level allowlist.
        allowed_types = relationship_types if relationship_types else _USER_FACING_REL_TYPES

        # C1 currency on every hop: clamped history copies and RETRACT
        # tombstones carry is_latest_belief=true with a past/empty valid_to,
        # so the interval arms are load-bearing here, not defensive.
        query = f"""
        MATCH path = (start:__Entity__ {{id: $entity_id}})-[*1..{max_hops}]-(related:__Entity__)
        WHERE ALL(node IN nodes(path) WHERE node:__Entity__)
          AND ALL(rel IN relationships(path)
                  WHERE type(rel) IN $allowed_rel_types
                    AND (rel.status IS NULL OR rel.status <> 'orphaned')
                    AND coalesce(rel.is_latest_belief, true)
                    AND (rel.valid_to IS NULL OR rel.valid_to > $now)
                    AND (rel.valid_from IS NULL OR rel.valid_from = '-inf'
                         OR rel.valid_from <= $now))
        WITH path, relationships(path) as rels, nodes(path) as nodes
        UNWIND range(0, size(rels)-1) as idx
        RETURN
            size(rels) as path_length,
            nodes[idx].id as source,
            nodes[idx].entity_type as source_type,
            type(rels[idx]) as relationship,
            nodes[idx+1].id as target,
            nodes[idx+1].entity_type as target_type,
            properties(rels[idx]) as properties
        """

        params: dict[str, Any] = {
            "entity_id": entity_id,
            "allowed_rel_types": allowed_types,
            "now": datetime.now(UTC).isoformat(),
        }
        results = self.connection.execute_query(query, params)

        return [dict(record) for record in results]

    def get_user_relationships_to_entities(
        self, user_id: str, entity_ids: list[str], relationship_types: list[str] | None = None
    ) -> list[dict]:
        """Get all relationships between User and specific entities.

        This finds direct connections: User -[r]-> Entity or User <-[r]- Entity

        Args:
            user_id: User entity ID (typically "User")
            entity_ids: List of entity IDs to check connections to
            relationship_types: Optional filter for specific relationships

        Returns:
            List of relationship dicts
        """
        # ADR-009 v1.1: always enforce user-facing rel-type allowlist.
        allowed_types = relationship_types if relationship_types else _USER_FACING_REL_TYPES

        # C1 currency: latest belief AND valid now. The valid_from arm keeps
        # future-dated facts out of "currently true" reads (design 11); '-inf'
        # is the ALWAYS sentinel.
        query = """
        MATCH (user:__Entity__ {id: $user_id})-[r]-(entity:__Entity__)
        WHERE entity.id IN $entity_ids
          AND type(r) IN $allowed_rel_types
          AND (r.status IS NULL OR r.status <> 'orphaned')
          AND coalesce(r.is_latest_belief, true)
          AND (r.valid_to IS NULL OR r.valid_to > $now)
          AND (r.valid_from IS NULL OR r.valid_from = '-inf' OR r.valid_from <= $now)
        RETURN
            user.id as user_id,
            entity.id as entity_id,
            entity.entity_type as entity_type,
            type(r) as relationship_type,
            properties(r) as properties,
            CASE
                WHEN startNode(r) = user THEN 'outgoing'
                ELSE 'incoming'
            END as direction
        """

        params: dict[str, Any] = {
            "user_id": user_id,
            "entity_ids": entity_ids,
            "allowed_rel_types": allowed_types,
            "now": datetime.now(UTC).isoformat(),
        }

        results = self.connection.execute_query(query, params)
        return [dict(record) for record in results]

    def get_all_user_relationships(
        self,
        user_id: str,
        relationship_types: list[str] | None = None,
        entity_types: list[str] | None = None,
    ) -> list[dict]:
        """Get ALL relationships from User entity.

        Useful for "What do I know?" type queries.

        Args:
            user_id: User entity ID
            relationship_types: Optional filter for specific relationships
            entity_types: Optional filter for specific entity types

        Returns:
            List of relationship dicts
        """
        # ADR-009 v1.1: always enforce user-facing rel-type allowlist.
        allowed_types = relationship_types if relationship_types else _USER_FACING_REL_TYPES

        filters: list[str] = [
            "type(r) IN $allowed_rel_types",
            "(r.status IS NULL OR r.status <> 'orphaned')",
            # C1 currency (latest belief, valid now; '-inf' = ALWAYS sentinel).
            "coalesce(r.is_latest_belief, true)",
            "(r.valid_to IS NULL OR r.valid_to > $now)",
            "(r.valid_from IS NULL OR r.valid_from = '-inf' OR r.valid_from <= $now)",
        ]
        if entity_types:
            filters.append("entity.entity_type IN $entity_types")

        where_clause = f"WHERE {' AND '.join(filters)}"

        query = f"""
        MATCH (user:__Entity__ {{id: $user_id}})-[r]->(entity:__Entity__)
        {where_clause}
        RETURN
            entity.id as entity_id,
            entity.entity_type as entity_type,
            type(r) as relationship_type,
            properties(r) as properties
        ORDER BY entity.entity_type, entity.id
        """

        params: dict[str, Any] = {
            "user_id": user_id,
            "allowed_rel_types": allowed_types,
            "now": datetime.now(UTC).isoformat(),
        }
        if entity_types:
            params["entity_types"] = entity_types
        results = self.connection.execute_query(query, params)
        return [dict(record) for record in results]

    def ensure_mist_identity(self) -> None:
        """Create the MistIdentity singleton node if it does not exist.

        MistIdentity is the hub of MIST's self-model. All internal entities
        (MistTrait, MistCapability, MistPreference, MistUncertainty) link
        to it via HAS_TRAIT, HAS_CAPABILITY, HAS_PREFERENCE, IS_UNCERTAIN_ABOUT.
        """
        query = """
        MERGE (m:__Entity__:MistIdentity {id: 'mist-identity'})
        ON CREATE SET
            m.entity_type = 'MistIdentity',
            m.display_name = 'MIST',
            m.knowledge_domain = 'internal',
            m.personality_summary = 'A cognitive architecture with persistent memory.',
            m.confidence = 1.0,
            m.status = 'active',
            m.created_at = datetime(),
            m.ontology_version = '1.0.0'
        """
        self.connection.execute_write(query)
        logger.debug("MistIdentity singleton ensured")

    def get_mist_identity_context(self) -> dict:
        """Fetch MIST identity + traits + capabilities + preferences from the graph.

        Returns a dict with keys:
            identity (single dict with id/display_name/pronouns/self_concept),
            traits (list of dicts),
            capabilities (list of dicts),
            preferences (list of dicts).

        Each trait / capability / preference row carries an `origin` field
        of "seeded" or "extracted":

        - "seeded": the canonical seed path documented in ADR-008. MistIdentity
          -[HAS_TRAIT]-> MistTrait (internal domain), MistIdentity
          -[HAS_CAPABILITY]-> MistCapability, MistIdentity
          -[HAS_PREFERENCE]-> MistPreference. These carry explicit `axis`
          (traits) and `enforcement` (preferences, including the "absolute"
          values pref-no-emoji / pref-no-ai-slop that drive the HARD RULES
          framing in MistContext.as_system_prompt_block).

        - "extracted": the Cluster 1 external path. MistIdentity
          -[MIST_HAS_TRAIT]-> Concept / Topic / Skill / Preference, MistIdentity
          -[MIST_HAS_CAPABILITY]-> Technology / Skill / Concept / Topic,
          MistIdentity -[MIST_HAS_PREFERENCE]-> Preference / Concept /
          Technology / Topic. Extracted targets use the external-entity
          property set (display_name, description) and are mapped into the
          renderer shape with `axis` defaulted to "Persona" and `enforcement`
          defaulted to "informational" so they are NEVER surfaced as HARD
          RULES -- absolute-enforcement prefs are seed-only by policy.

        Used by KnowledgeRetriever.retrieve_mist_context() for persona injection.
        Traversal is anchored at the mist-identity node across both paths.
        Falls back to a minimal default identity dict when the node is absent
        (pre-seed state).
        """
        identity_query = """
            MATCH (m:__Entity__ {id: 'mist-identity'})
            RETURN m.id AS id, m.display_name AS display_name,
                   m.pronouns AS pronouns, m.self_concept AS self_concept
        """
        # Seeded path: HAS_TRAIT/HAS_CAPABILITY/HAS_PREFERENCE -> internal
        # MistTrait / MistCapability / MistPreference nodes with canonical shape.
        # C1 currency filter on every persona edge: latest belief AND valid
        # now ('-inf' = ALWAYS sentinel; legacy unstamped edges coalesce true).
        # Orphan arm is the ONLY retirement channel for persona edges: HAS_*
        # is not extractable, so vault user-edits retire them via
        # status='orphaned' (rebuild_from_path) and nothing interval-closes
        # them.
        currency = """WHERE (r.status IS NULL OR r.status <> 'orphaned')
              AND coalesce(r.is_latest_belief, true)
              AND (r.valid_to IS NULL OR r.valid_to > $now)
              AND (r.valid_from IS NULL OR r.valid_from = '-inf' OR r.valid_from <= $now)"""
        seeded_traits_query = f"""
            MATCH (m:__Entity__ {{id: 'mist-identity'}})-[r:HAS_TRAIT]->(t:__Entity__)
            {currency}
            RETURN t.id AS id, t.display_name AS display_name,
                   t.axis AS axis, t.description AS description,
                   t.entity_type AS entity_type
            ORDER BY t.display_name
        """
        seeded_capabilities_query = f"""
            MATCH (m:__Entity__ {{id: 'mist-identity'}})-[r:HAS_CAPABILITY]->(c:__Entity__)
            {currency}
            RETURN c.id AS id, c.display_name AS display_name,
                   c.description AS description,
                   c.entity_type AS entity_type
            ORDER BY c.display_name
        """
        seeded_preferences_query = f"""
            MATCH (m:__Entity__ {{id: 'mist-identity'}})-[r:HAS_PREFERENCE]->(p:__Entity__)
            {currency}
            RETURN p.id AS id, p.display_name AS display_name,
                   p.enforcement AS enforcement, p.context AS context,
                   p.entity_type AS entity_type
            ORDER BY p.enforcement DESC, p.display_name
        """
        # Extracted path (Cluster 1): MIST_HAS_TRAIT / MIST_HAS_CAPABILITY /
        # MIST_HAS_PREFERENCE -> external-domain nodes. These have display_name
        # and description only; axis/enforcement/context are synthesized below.
        extracted_traits_query = f"""
            MATCH (m:__Entity__ {{id: 'mist-identity'}})-[r:MIST_HAS_TRAIT]->(t:__Entity__)
            {currency}
            RETURN t.id AS id, t.display_name AS display_name,
                   t.description AS description,
                   t.entity_type AS entity_type
            ORDER BY t.display_name
        """
        extracted_capabilities_query = f"""
            MATCH (m:__Entity__ {{id: 'mist-identity'}})-[r:MIST_HAS_CAPABILITY]->(c:__Entity__)
            {currency}
            RETURN c.id AS id, c.display_name AS display_name,
                   c.description AS description,
                   c.entity_type AS entity_type
            ORDER BY c.display_name
        """
        extracted_preferences_query = f"""
            MATCH (m:__Entity__ {{id: 'mist-identity'}})-[r:MIST_HAS_PREFERENCE]->(p:__Entity__)
            {currency}
            RETURN p.id AS id, p.display_name AS display_name,
                   p.description AS description,
                   p.entity_type AS entity_type
            ORDER BY p.display_name
        """
        # IMPLEMENTED_WITH is a MIST-scope edge that also surfaces as a
        # capability-flavored fact ("MIST is implemented with X"). Project
        # it into the capabilities list so the persona block covers
        # implementation-stack facts.
        implemented_with_query = f"""
            MATCH (m:__Entity__ {{id: 'mist-identity'}})-[r:IMPLEMENTED_WITH]->(t:__Entity__)
            {currency}
            RETURN t.id AS id, t.display_name AS display_name,
                   t.description AS description,
                   t.entity_type AS entity_type
            ORDER BY t.display_name
        """

        now_params = {"now": datetime.now(UTC).isoformat()}
        identity_rows = self.connection.execute_query(identity_query, {})
        seeded_trait_rows = self.connection.execute_query(seeded_traits_query, now_params)
        seeded_capability_rows = self.connection.execute_query(
            seeded_capabilities_query, now_params
        )
        seeded_preference_rows = self.connection.execute_query(seeded_preferences_query, now_params)
        extracted_trait_rows = self.connection.execute_query(extracted_traits_query, now_params)
        extracted_capability_rows = self.connection.execute_query(
            extracted_capabilities_query, now_params
        )
        extracted_preference_rows = self.connection.execute_query(
            extracted_preferences_query, now_params
        )
        implemented_with_rows = self.connection.execute_query(implemented_with_query, now_params)

        identity = (
            identity_rows[0]
            if identity_rows
            else {
                "id": "mist-identity",
                "display_name": "MIST",
                "pronouns": "she/her",
                "self_concept": "",
            }
        )

        traits: list[dict] = [{**dict(r), "origin": "seeded"} for r in seeded_trait_rows]
        for row in extracted_trait_rows:
            entry = dict(row)
            traits.append(
                {
                    "id": entry.get("id"),
                    "display_name": entry.get("display_name"),
                    # Default axis for extracted traits -- not a seed with a
                    # canonical persona/platform distinction.
                    "axis": "Persona",
                    "description": entry.get("description") or "",
                    "entity_type": entry.get("entity_type"),
                    "origin": "extracted",
                }
            )

        capabilities: list[dict] = [{**dict(r), "origin": "seeded"} for r in seeded_capability_rows]
        for row in extracted_capability_rows:
            entry = dict(row)
            capabilities.append(
                {
                    "id": entry.get("id"),
                    "display_name": entry.get("display_name"),
                    "description": entry.get("description") or "",
                    "entity_type": entry.get("entity_type"),
                    "origin": "extracted",
                }
            )
        for row in implemented_with_rows:
            entry = dict(row)
            name = entry.get("display_name") or ""
            desc = entry.get("description") or ""
            # Synthesize a capability-flavored description so the renderer
            # surfaces "implemented with" facts naturally.
            synth_desc = f"Implemented with {name}." if name else desc
            if desc and name and name not in desc:
                synth_desc = f"{synth_desc} {desc}"
            capabilities.append(
                {
                    "id": entry.get("id"),
                    "display_name": name,
                    "description": synth_desc.strip(),
                    "entity_type": entry.get("entity_type"),
                    "origin": "extracted",
                }
            )

        preferences: list[dict] = [{**dict(r), "origin": "seeded"} for r in seeded_preference_rows]
        for row in extracted_preference_rows:
            entry = dict(row)
            preferences.append(
                {
                    "id": entry.get("id"),
                    "display_name": entry.get("display_name"),
                    # Absolute enforcement is seed-only -- extracted prefs default
                    # to informational so HARD RULES stay gated on seed policy.
                    "enforcement": "informational",
                    "context": entry.get("description") or "",
                    "entity_type": entry.get("entity_type"),
                    "origin": "extracted",
                }
            )

        return {
            "identity": dict(identity),
            "traits": traits,
            "capabilities": capabilities,
            "preferences": preferences,
        }

    # ------------------------------------------------------------------
    # GraphStoreProtocol methods (ADR-010 / Phase 5 reviewer P0)
    # ------------------------------------------------------------------

    def current_ontology_version(self) -> str:
        """Return the current ontology version string (e.g. '1.1.0').

        Synchronous accessor used by GraphRegenerator to stamp re-derived
        triples with the version active at rebuild time. The value is set
        at construction by the factory (injected from KnowledgeConfig) and
        never changes during the lifetime of this store instance.
        """
        return self._ontology_version

    async def mark_orphaned_by_provenance_path(self, path: str) -> int:
        """Mark all triples with DERIVED_FROM.path == path as status='orphaned'.

        Per ADR-010 invariant 5: preserves the triples (no hard-delete) so
        they can be re-derived on retry. Marks both the DERIVED_FROM edge
        itself and the typed edges whose `derived_from_path` property matches
        the path, so retrieval queries can filter on the typed edge's own
        status field without a JOIN through DERIVED_FROM per query.

        Two Cypher writes are issued:
        1. Mark DERIVED_FROM edges pointing at the VaultNote for path.
        2. Mark any typed edge whose r.derived_from_path == path (the
           property stamped by upsert_identity / upsert_user at write time).

        Both writes are idempotent: the WHERE predicate excludes already-
        orphaned edges so re-running with the same path returns 0 on the
        second call.

        Returns:
            Total count of edges marked (DERIVED_FROM edges + typed edges).
        """
        params: dict[str, Any] = {"path": path}
        loop = asyncio.get_event_loop()

        # --- Write 1: mark DERIVED_FROM provenance edges ---
        # coalesce: legacy DERIVED_FROM edges predate the status property;
        # under three-valued logic a bare `d.status <> 'orphaned'` filters
        # null-status edges OUT and the orphan-marking silently no-ops.
        derived_from_query = (
            "MATCH ()-[d:DERIVED_FROM]->(p:__Provenance__:VaultNote {path: $path}) "
            "WHERE coalesce(d.status, 'active') <> 'orphaned' "
            "SET d.status = 'orphaned', d.orphaned_at = toString(datetime()) "
            "RETURN count(d) AS marked_count"
        )
        rows_df = await loop.run_in_executor(
            None, self.connection.execute_write, derived_from_query, params
        )
        d_count = int(rows_df[0].get("marked_count", 0)) if rows_df else 0

        # --- Write 2: mark typed edges by derived_from_path property ---
        # Scoped to the known user-facing relationship types (the allowlist
        # mirrors _USER_FACING_REL_TYPES) so the MATCH is not a full graph
        # scan. The WHERE clause is defensive against already-orphaned edges
        # to preserve idempotency on repeat calls.
        typed_edge_query = (
            "MATCH ()-[r]->() "
            "WHERE r.derived_from_path = $path "
            "  AND (r.status IS NULL OR r.status <> 'orphaned') "
            "SET r.status = 'orphaned', r.orphaned_at = toString(datetime()) "
            "RETURN count(r) AS marked_count"
        )
        rows_te = await loop.run_in_executor(
            None, self.connection.execute_write, typed_edge_query, params
        )
        r_count = int(rows_te[0].get("marked_count", 0)) if rows_te else 0

        return d_count + r_count

    async def get_orphaned_provenance_paths(self) -> list[str]:
        """Return the list of distinct DERIVED_FROM.path values for orphaned triples.

        Used by GraphRegenerator.retry_orphaned to enumerate provenance paths
        whose async re-extraction previously failed.

        Returns:
            List of absolute path strings for which orphaned triples exist.
        """
        query = (
            "MATCH ()-[d:DERIVED_FROM]->(p:__Provenance__:VaultNote) "
            "WHERE d.status = 'orphaned' "
            "RETURN DISTINCT p.path AS path"
        )
        loop = asyncio.get_event_loop()
        rows = await loop.run_in_executor(None, self.connection.execute_query, query, None)
        return [r.get("path") for r in rows if r.get("path") is not None]

    async def upsert_identity(
        self,
        parsed_identity: "ParsedIdentity",
        derived_from_path: str,
    ) -> int:
        """Upsert MistIdentity attributes from a ParsedIdentity (idempotent).

        Writes traits, capabilities, and preferences to the graph as typed
        edges from the MistIdentity node. Each edge gets a fresh DERIVED_FROM
        edge to the VaultNote identified by `derived_from_path`. Uses MERGE
        throughout so re-running with the same input produces no duplicates.

        Returns:
            Count of new triples written (0 when all already exist).
        """
        now = datetime.utcnow().isoformat()
        path = derived_from_path
        ontology_version = self._ontology_version

        # Ensure the provenance VaultNote node exists
        self.connection.execute_write(
            "MERGE (vn:__Provenance__:VaultNote {path: $path}) "
            "ON CREATE SET vn.created_at = $now, vn.status = 'active' "
            "ON MATCH SET vn.updated_at = $now",
            {"path": path, "now": now},
        )

        written = 0

        # HAS_TRAIT edges
        for slug in parsed_identity.traits:
            entity_id = f"mist-trait-{slug}"
            self.connection.execute_write(
                "MERGE (m:__Entity__:MistIdentity {id: 'mist-identity'}) "
                "MERGE (t:__Entity__:MistTrait {id: $entity_id}) "
                "ON CREATE SET t.display_name = $slug, t.entity_type = 'MistTrait', "
                "t.status = 'active', t.ontology_version = $ontology_version "
                "MERGE (m)-[r:HAS_TRAIT]->(t) "
                "ON CREATE SET r.derived_from_path = $path, r.created_at = $now, "
                "r.status = 'active', r.ontology_version = $ontology_version "
                "ON MATCH SET r.derived_from_path = $path, r.updated_at = $now, "
                "r.status = 'active', r.ontology_version = $ontology_version",
                {
                    "entity_id": entity_id,
                    "slug": slug,
                    "path": path,
                    "now": now,
                    "ontology_version": ontology_version,
                },
            )
            # DERIVED_FROM provenance edge: typed entity -> VaultNote.
            # mark_orphaned_by_provenance_path queries this relationship-type,
            # so Bucket 1 edges must carry it to be findable on user-edit.
            self.connection.execute_write(
                "MATCH (t:__Entity__:MistTrait {id: $entity_id}) "
                "MATCH (vn:__Provenance__:VaultNote {path: $path}) "
                "MERGE (t)-[df:DERIVED_FROM]->(vn) "
                "ON CREATE SET df.created_at = $now, "
                "df.ontology_version = $ontology_version, df.status = 'active' "
                "ON MATCH SET df.status = 'active'",
                {
                    "entity_id": entity_id,
                    "path": path,
                    "now": now,
                    "ontology_version": ontology_version,
                },
            )
            written += 1

        # HAS_CAPABILITY edges
        for slug in parsed_identity.capabilities:
            entity_id = f"mist-cap-{slug}"
            self.connection.execute_write(
                "MERGE (m:__Entity__:MistIdentity {id: 'mist-identity'}) "
                "MERGE (c:__Entity__:MistCapability {id: $entity_id}) "
                "ON CREATE SET c.display_name = $slug, c.entity_type = 'MistCapability', "
                "c.status = 'active', c.ontology_version = $ontology_version "
                "MERGE (m)-[r:HAS_CAPABILITY]->(c) "
                "ON CREATE SET r.derived_from_path = $path, r.created_at = $now, "
                "r.status = 'active', r.ontology_version = $ontology_version "
                "ON MATCH SET r.derived_from_path = $path, r.updated_at = $now, "
                "r.status = 'active', r.ontology_version = $ontology_version",
                {
                    "entity_id": entity_id,
                    "slug": slug,
                    "path": path,
                    "now": now,
                    "ontology_version": ontology_version,
                },
            )
            # DERIVED_FROM provenance edge: typed entity -> VaultNote.
            self.connection.execute_write(
                "MATCH (c:__Entity__:MistCapability {id: $entity_id}) "
                "MATCH (vn:__Provenance__:VaultNote {path: $path}) "
                "MERGE (c)-[df:DERIVED_FROM]->(vn) "
                "ON CREATE SET df.created_at = $now, "
                "df.ontology_version = $ontology_version, df.status = 'active' "
                "ON MATCH SET df.status = 'active'",
                {
                    "entity_id": entity_id,
                    "path": path,
                    "now": now,
                    "ontology_version": ontology_version,
                },
            )
            written += 1

        # HAS_PREFERENCE edges
        for pref in parsed_identity.preferences:
            entity_id = f"mist-pref-{pref.slug}"
            self.connection.execute_write(
                "MERGE (m:__Entity__:MistIdentity {id: 'mist-identity'}) "
                "MERGE (p:__Entity__:MistPreference {id: $entity_id}) "
                "ON CREATE SET p.display_name = $slug, p.entity_type = 'MistPreference', "
                "p.enforcement = $enforcement, p.status = 'active', "
                "p.ontology_version = $ontology_version "
                "ON MATCH SET p.enforcement = $enforcement "
                "MERGE (m)-[r:HAS_PREFERENCE]->(p) "
                "ON CREATE SET r.derived_from_path = $path, r.created_at = $now, "
                "r.status = 'active', r.ontology_version = $ontology_version "
                "ON MATCH SET r.derived_from_path = $path, r.updated_at = $now, "
                "r.status = 'active', r.ontology_version = $ontology_version",
                {
                    "entity_id": entity_id,
                    "slug": pref.slug,
                    "enforcement": pref.enforcement,
                    "path": path,
                    "now": now,
                    "ontology_version": ontology_version,
                },
            )
            # DERIVED_FROM provenance edge: typed entity -> VaultNote.
            self.connection.execute_write(
                "MATCH (p:__Entity__:MistPreference {id: $entity_id}) "
                "MATCH (vn:__Provenance__:VaultNote {path: $path}) "
                "MERGE (p)-[df:DERIVED_FROM]->(vn) "
                "ON CREATE SET df.created_at = $now, "
                "df.ontology_version = $ontology_version, df.status = 'active' "
                "ON MATCH SET df.status = 'active'",
                {
                    "entity_id": entity_id,
                    "path": path,
                    "now": now,
                    "ontology_version": ontology_version,
                },
            )
            written += 1

        logger.debug(
            "upsert_identity: wrote %d triples from %s (ontology %s)",
            written,
            path,
            ontology_version,
        )
        return written

    async def upsert_user(
        self,
        parsed_user: "ParsedUser",
        derived_from_path: str,
    ) -> int:
        """Upsert User neighbor edges from a ParsedUser (idempotent).

        Writes per-section display_name lists as typed edges from the User
        node identified by `parsed_user.user_id`. Each edge gets a fresh
        DERIVED_FROM provenance stamp to the VaultNote at `derived_from_path`.
        Uses MERGE so re-running with the same input produces no duplicates.

        Edge types map to bucket1_reader section labels:
            tools_and_technologies -> USES
            expertise              -> EXPERT_IN
            currently_learning     -> LEARNING
            projects               -> WORKS_ON
            affiliations           -> WORKS_AT
            interests              -> INTERESTED_IN
            goals                  -> HAS_GOAL
            preferences            -> PREFERS
            people                 -> KNOWS_PERSON

        Returns:
            Count of new triples written (0 when all already exist).
        """
        now = datetime.utcnow().isoformat()
        path = derived_from_path
        ontology_version = self._ontology_version
        user_id = parsed_user.user_id

        # Ensure the provenance VaultNote node exists
        self.connection.execute_write(
            "MERGE (vn:__Provenance__:VaultNote {path: $path}) "
            "ON CREATE SET vn.created_at = $now, vn.status = 'active' "
            "ON MATCH SET vn.updated_at = $now",
            {"path": path, "now": now},
        )

        section_map: list[tuple[str, list[str]]] = [
            ("USES", parsed_user.tools_and_technologies),
            ("EXPERT_IN", parsed_user.expertise),
            ("LEARNING", parsed_user.currently_learning),
            ("WORKS_ON", parsed_user.projects),
            ("WORKS_AT", parsed_user.affiliations),
            ("INTERESTED_IN", parsed_user.interests),
            ("HAS_GOAL", parsed_user.goals),
            ("PREFERS", parsed_user.preferences),
            ("KNOWS_PERSON", parsed_user.people),
        ]

        written = 0
        for predicate, targets in section_map:
            for display_name in targets:
                target_id = f"entity-{display_name.lower().replace(' ', '-')}"
                self.connection.execute_write(
                    # Label-safe user MERGE: match on :__Entity__ {id} and SET
                    # the :User label, so a label-less extraction-created user
                    # node is healed instead of tripping entity_id_unique
                    # (deep review cypher-data-integrity-2b).
                    f"MERGE (u:__Entity__ {{id: $user_id}}) "
                    f"SET u:User "
                    f"MERGE (t:__Entity__ {{id: $target_id}}) "
                    f"ON CREATE SET t.display_name = $display_name, "
                    f"t.status = 'active', t.ontology_version = $ontology_version "
                    f"MERGE (u)-[r:{predicate}]->(t) "
                    f"ON CREATE SET r.derived_from_path = $path, r.created_at = $now, "
                    f"r.status = 'active', r.ontology_version = $ontology_version "
                    f"ON MATCH SET r.derived_from_path = $path, r.updated_at = $now, "
                    f"r.status = 'active', r.ontology_version = $ontology_version",
                    {
                        "user_id": user_id,
                        "target_id": target_id,
                        "display_name": display_name,
                        "path": path,
                        "now": now,
                        "ontology_version": ontology_version,
                    },
                )
                # DERIVED_FROM provenance edge: target entity -> VaultNote.
                # mark_orphaned_by_provenance_path queries this relationship-type,
                # so Bucket 1 edges must carry it to be findable on user-edit.
                self.connection.execute_write(
                    "MATCH (t:__Entity__ {id: $target_id}) "
                    "MATCH (vn:__Provenance__:VaultNote {path: $path}) "
                    "MERGE (t)-[df:DERIVED_FROM]->(vn) "
                    "ON CREATE SET df.created_at = $now, "
                    "df.ontology_version = $ontology_version, df.status = 'active' "
                    "ON MATCH SET df.status = 'active'",
                    {
                        "target_id": target_id,
                        "path": path,
                        "now": now,
                        "ontology_version": ontology_version,
                    },
                )
                written += 1

        logger.debug(
            "upsert_user: wrote %d triples for user %s from %s (ontology %s)",
            written,
            user_id,
            path,
            ontology_version,
        )
        return written

    def close(self):
        """Close Neo4j connection."""
        self.connection.disconnect()
