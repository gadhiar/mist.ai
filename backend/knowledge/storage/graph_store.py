"""
Graph Storage Module

Stores extracted entities and relationships in Neo4j with provenance tracking.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid
import logging

from backend.knowledge.config import KnowledgeConfig
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
            "CREATE CONSTRAINT utterance_id_unique IF NOT EXISTS FOR (u:Utterance) REQUIRE u.utterance_id IS UNIQUE"
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
            "CREATE INDEX utterance_timestamp_idx IF NOT EXISTS FOR (u:Utterance) ON (u.timestamp)"
        ]

        for index in indexes:
            try:
                self.connection.execute_write(index)
                logger.debug(f"Created index: {index[:50]}...")
            except Exception as e:
                logger.warning(f"Index may already exist: {e}")

        logger.info("Schema initialization complete")

    def store_conversation_event(
        self,
        conversation_id: str,
        user_id: str,
        timestamp: Optional[datetime] = None
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
            "timestamp": timestamp.isoformat()
        }

        result = self.connection.execute_query(query, params)
        return result[0]["id"] if result else conversation_id

    def store_utterance(
        self,
        utterance_id: str,
        conversation_id: str,
        text: str,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
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
                "metadata": metadata
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
                "timestamp": timestamp.isoformat()
            }

        result = self.connection.execute_query(query, params)
        return result[0]["id"] if result else utterance_id

    def store_extracted_entities(
        self,
        graph_document,
        utterance_id: str,
        ontology_version: Optional[str] = None
    ):
        """
        Store extracted entities and relationships from LLMGraphTransformer

        Args:
            graph_document: GraphDocument from LLMGraphTransformer
            utterance_id: Source utterance ID
            ontology_version: Ontology version used for extraction
        """
        if ontology_version is None:
            ontology_version = self.config.ontology_version

        logger.info(f"Storing {len(graph_document.nodes)} nodes and {len(graph_document.relationships)} relationships")

        # Store nodes
        for node in graph_document.nodes:
            self._store_node(node, utterance_id, ontology_version)

        # Store relationships
        for relationship in graph_document.relationships:
            self._store_relationship(relationship, utterance_id, ontology_version)

    def _store_node(self, node, utterance_id: str, ontology_version: str):
        """Store a single entity node"""

        # Extract node properties
        node_id = node.id
        entity_type = node.type if hasattr(node, 'type') else "Unknown"
        properties = getattr(node, 'properties', {})

        # Build query with dynamic property setting
        # Neo4j doesn't allow nested dictionaries, so flatten properties
        property_sets = []
        params = {
            "utterance_id": utterance_id,
            "node_id": node_id,
            "entity_type": entity_type,
            "ontology_version": ontology_version
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
            e.created_at = datetime(),
            e.created_from_utterance = $utterance_id"""

        if property_sets:
            all_sets = base_sets + ",\n            " + ",\n            ".join(property_sets)
        else:
            all_sets = base_sets

        query = f"""
        MATCH (u:Utterance {{utterance_id: $utterance_id}})
        MERGE (e:__Entity__ {{id: $node_id}})
        ON CREATE SET
            {all_sets}
        MERGE (u)-[:HAS_ENTITY]->(e)
        """

        self.connection.execute_write(query, params)
        logger.debug(f"Stored node: {node_id} ({entity_type})")

    def _store_relationship(self, relationship, utterance_id: str, ontology_version: str):
        """Store a single relationship between entities"""

        # Extract relationship details
        source_id = relationship.source.id
        target_id = relationship.target.id
        rel_type = relationship.type
        properties = getattr(relationship, 'properties', {})

        # Sanitize relationship type for Cypher (no spaces, special chars)
        rel_type_safe = rel_type.replace(" ", "_").replace("-", "_").upper()

        # Build query with dynamic property setting
        property_sets = []
        params = {
            "source_id": source_id,
            "target_id": target_id,
            "ontology_version": ontology_version,
            "utterance_id": utterance_id
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
            r.created_at = datetime(),
            r.created_from_utterance = $utterance_id"""

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
        logger.debug(f"Stored relationship: {source_id} -[{rel_type}]-> {target_id}")

    def get_entities_for_conversation(self, conversation_id: str) -> List[Dict]:
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

    def close(self):
        """Close Neo4j connection"""
        self.connection.disconnect()
