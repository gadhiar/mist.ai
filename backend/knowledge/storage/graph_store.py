"""
Graph Storage Module

Stores extracted entities and relationships in Neo4j with provenance tracking.
"""

import logging
from datetime import datetime
from typing import Any

from backend.knowledge.config import KnowledgeConfig
from backend.knowledge.embeddings import EmbeddingGenerator
from backend.knowledge.storage.neo4j_connection import Neo4jConnection

logger = logging.getLogger(__name__)


class GraphStore:
    """
    Manages storage of knowledge graph in Neo4j

    Handles:
    - Storing extracted entities and relationships
    - Provenance tracking (which conversation created which entity)
    - Versioning support (ontology versions)
    """

    def __init__(self, config: KnowledgeConfig):
        """
        Initialize graph store

        Args:
            config: Knowledge system configuration
        """
        self.config = config
        self.connection = Neo4jConnection(config.neo4j)
        self.embedding_generator = EmbeddingGenerator()

    def initialize_schema(self):
        """
        Create indexes and constraints in Neo4j

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
        ]

        for constraint in constraints:
            try:
                self.connection.execute_write(constraint)
                logger.debug(f"Created constraint: {constraint[:50]}...")
            except Exception as e:
                logger.warning(f"Constraint may already exist: {e}")

        # Create indexes for performance
        indexes = [
            "CREATE INDEX entity_type_idx IF NOT EXISTS FOR (e:__Entity__) ON (e.entity_type)",
            "CREATE INDEX conversation_timestamp_idx IF NOT EXISTS FOR (c:ConversationEvent) ON (c.timestamp)",
            "CREATE INDEX utterance_timestamp_idx IF NOT EXISTS FOR (u:Utterance) ON (u.timestamp)",
            "CREATE INDEX source_type_idx IF NOT EXISTS FOR (s:SourceDocument) ON (s.source_type)",
            "CREATE INDEX chunk_position_idx IF NOT EXISTS FOR (c:DocumentChunk) ON (c.position)",
            "CREATE INDEX source_hash_idx IF NOT EXISTS FOR (s:SourceDocument) ON (s.content_hash)",
        ]

        for index in indexes:
            try:
                self.connection.execute_write(index)
                logger.debug(f"Created index: {index[:50]}...")
            except Exception as e:
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

        for vector_index in vector_indexes:
            try:
                self.connection.execute_write(vector_index)
                logger.info("Created vector index")
            except Exception as e:
                logger.warning(f"Vector index creation failed (may not be supported): {e}")
                logger.warning("Vector search will not be available without Neo4j 5.11+")

        logger.info("Schema initialization complete")

    def store_conversation_event(
        self, conversation_id: str, user_id: str, timestamp: datetime | None = None
    ) -> str:
        """
        Store a conversation event

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

        result = self.connection.execute_query(query, params)
        return result[0]["id"] if result else conversation_id

    def store_utterance(
        self,
        utterance_id: str,
        conversation_id: str,
        text: str,
        timestamp: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Store a single utterance

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

        result = self.connection.execute_query(query, params)
        return result[0]["id"] if result else utterance_id

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
        """
        Store a source document

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

        result = self.connection.execute_query(query, params)
        return result[0]["id"] if result else source_id

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
        """
        Store a document chunk with optional embedding

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

        result = self.connection.execute_query(query, params)
        return result[0]["id"] if result else chunk_id

    def store_extracted_entities(
        self,
        graph_document,
        utterance_id: str | None = None,
        chunk_id: str | None = None,
        ontology_version: str | None = None,
    ):
        """
        Store extracted entities and relationships from LLMGraphTransformer

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

        if ontology_version is None:
            ontology_version = self.config.ontology_version

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

    def _store_node(self, node, source_id: str, source_type: str, ontology_version: str):
        """
        Store a single entity node with provenance

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
        # Use entity_id as the text to embed (could be enhanced with properties)
        embedding = self.embedding_generator.generate_embedding(node_id)

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

        # Add each property individually if they exist
        if properties:
            for key, value in properties.items():
                # Only add primitive types (string, number, boolean)
                if isinstance(value, (str, int, float, bool)):
                    param_key = f"prop_{key}"
                    property_sets.append(f"e.{key} = ${param_key}")
                    params[param_key] = value

        # Build the SET clause
        base_sets = """
            e.entity_type = $entity_type,
            e.ontology_version = $ontology_version,
            e.embedding = $embedding,
            e.created_at = datetime()"""

        if property_sets:
            all_sets = base_sets + ",\n            " + ",\n            ".join(property_sets)
        else:
            all_sets = base_sets

        # Different query based on source type
        if source_type == "utterance":
            query = f"""
            MATCH (u:Utterance {{utterance_id: $source_id}})
            MERGE (e:__Entity__ {{id: $node_id}})
            ON CREATE SET
                {all_sets}
            MERGE (u)-[:HAS_ENTITY]->(e)
            """
        else:  # chunk
            query = f"""
            MATCH (c:DocumentChunk {{chunk_id: $source_id}})
            MERGE (e:__Entity__ {{id: $node_id}})
            ON CREATE SET
                {all_sets}
            MERGE (e)-[:EXTRACTED_FROM]->(c)
            """

        self.connection.execute_write(query, params)
        logger.debug(f"Stored node: {node_id} ({entity_type}) from {source_type}")

    def _store_relationship(
        self, relationship, source_id: str, source_type: str, ontology_version: str
    ):
        """
        Store a single relationship between entities

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

        # Sanitize relationship type for Cypher (no spaces, special chars)
        rel_type_safe = rel_type.replace(" ", "_").replace("-", "_").upper()

        # Build query with dynamic property setting
        property_sets = []
        params = {
            "source_id": entity_source_id,
            "target_id": entity_target_id,
            "ontology_version": ontology_version,
        }

        # Add each property individually if they exist
        if properties:
            for key, value in properties.items():
                # Only add primitive types (string, number, boolean)
                if isinstance(value, (str, int, float, bool)):
                    param_key = f"prop_{key}"
                    property_sets.append(f"r.{key} = ${param_key}")
                    params[param_key] = value

        # Build the SET clause
        base_sets = """
            r.ontology_version = $ontology_version,
            r.created_at = datetime()"""

        if property_sets:
            all_sets = base_sets + ",\n            " + ",\n            ".join(property_sets)
        else:
            all_sets = base_sets

        query = f"""
        MATCH (source:__Entity__ {{id: $source_id}})
        MATCH (target:__Entity__ {{id: $target_id}})
        MERGE (source)-[r:{rel_type_safe}]->(target)
        ON CREATE SET
            {all_sets}
        """

        self.connection.execute_write(query, params)
        logger.debug(f"Stored relationship: {entity_source_id} -[{rel_type}]-> {entity_target_id}")

    def get_entities_for_conversation(self, conversation_id: str) -> list[dict]:
        """
        Retrieve all entities extracted from a conversation

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
        """
        Search for entities semantically similar to query text

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
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            logger.error("Ensure Neo4j 5.11+ is installed and vector index is created")
            raise

    def search_document_chunks(
        self, query_text: str, limit: int = 5, similarity_threshold: float = 0.7
    ) -> list[dict]:
        """
        Search DocumentChunks using vector similarity (RAG retrieval)

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
        except Exception as e:
            logger.error(f"Document chunk search failed: {e}")
            logger.error(
                "Ensure Neo4j 5.11+ is installed and chunk_embeddings vector index is created"
            )
            raise

    def get_entity_neighborhood(
        self, entity_id: str, max_hops: int = 2, relationship_types: list[str] | None = None
    ) -> list[dict]:
        """
        Get N-hop neighborhood around an entity

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
        if relationship_types:
            rel_filter = (
                f"WHERE ALL(r in relationships(path) WHERE type(r) IN {relationship_types})"
            )
        else:
            rel_filter = ""

        query = f"""
        MATCH path = (start:__Entity__ {{id: $entity_id}})-[*1..{max_hops}]-(related:__Entity__)
        {rel_filter}
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

        params = {"entity_id": entity_id}
        results = self.connection.execute_query(query, params)

        return [dict(record) for record in results]

    def get_user_relationships_to_entities(
        self, user_id: str, entity_ids: list[str], relationship_types: list[str] | None = None
    ) -> list[dict]:
        """
        Get all relationships between User and specific entities

        This finds direct connections: User -[r]-> Entity or User <-[r]- Entity

        Args:
            user_id: User entity ID (typically "User")
            entity_ids: List of entity IDs to check connections to
            relationship_types: Optional filter for specific relationships

        Returns:
            List of relationship dicts
        """
        rel_filter = f"AND type(r) IN {relationship_types}" if relationship_types else ""

        query = f"""
        MATCH (user:__Entity__ {{id: $user_id}})-[r]-(entity:__Entity__)
        WHERE entity.id IN $entity_ids {rel_filter}
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

        params = {"user_id": user_id, "entity_ids": entity_ids}

        results = self.connection.execute_query(query, params)
        return [dict(record) for record in results]

    def get_all_user_relationships(
        self,
        user_id: str,
        relationship_types: list[str] | None = None,
        entity_types: list[str] | None = None,
    ) -> list[dict]:
        """
        Get ALL relationships from User entity

        Useful for "What do I know?" type queries.

        Args:
            user_id: User entity ID
            relationship_types: Optional filter for specific relationships
            entity_types: Optional filter for specific entity types

        Returns:
            List of relationship dicts
        """
        filters = []
        if relationship_types:
            filters.append(f"type(r) IN {relationship_types}")
        if entity_types:
            filters.append(f"entity.entity_type IN {entity_types}")

        where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""

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

        params = {"user_id": user_id}
        results = self.connection.execute_query(query, params)
        return [dict(record) for record in results]

    def close(self):
        """Close Neo4j connection"""
        self.connection.disconnect()
