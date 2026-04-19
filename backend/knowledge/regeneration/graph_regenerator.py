"""Graph Regeneration Module.

Rebuilds the knowledge graph from immutable utterances.

This proves the architecture works:
- Utterances are the source of truth
- Knowledge graph is a materialized view
- Can always rebuild from scratch
"""

import logging
from datetime import datetime

from backend.knowledge.config import KnowledgeConfig
from backend.knowledge.extraction.pipeline import ExtractionPipeline
from backend.knowledge.models import RegenerationReport, Utterance
from backend.knowledge.storage.graph_store import GraphStore

logger = logging.getLogger(__name__)


class GraphRegenerator:
    """Regenerates knowledge graph from immutable utterances.

    The knowledge graph is a materialized view built from utterances.
    This class rebuilds the entire graph or specific conversations.

    Example:
        regenerator = GraphRegenerator(config)
        report = await regenerator.regenerate_all()
        print(f"Processed {report.processed} utterances")
    """

    def __init__(
        self,
        config: KnowledgeConfig,
        extraction_pipeline: ExtractionPipeline,
        graph_store: GraphStore,
    ) -> None:
        """Initialize graph regenerator.

        Args:
            config: Knowledge system configuration.
            extraction_pipeline: ExtractionPipeline (with include_curation=False).
            graph_store: Graph store for entity storage and graph operations.
        """
        self.config = config
        self._pipeline = extraction_pipeline
        self.graph_store = graph_store
        self.connection = graph_store.connection
        logger.info("GraphRegenerator initialized")

    async def regenerate_all(self) -> RegenerationReport:
        """Regenerate entire knowledge graph from all utterances.

        Process:
        1. Fetch all utterances from Neo4j
        2. Delete entity graph (preserve utterances)
        3. Re-extract entities from each utterance
        4. Store extracted entities
        5. Return statistics

        Returns:
            RegenerationReport with statistics

        Example:
            >>> report = await regenerator.regenerate_all()
            >>> print(f"Created {report.entities_created} entities")
        """
        start_time = datetime.now()

        logger.info("=" * 60)
        logger.info("Starting full graph regeneration")
        logger.info("=" * 60)

        try:
            # Step 1: Fetch utterances
            logger.info("Step 1: Fetching utterances...")
            utterances = self._fetch_all_utterances()
            logger.info(f"Found {len(utterances)} utterances to process")

            if len(utterances) == 0:
                logger.warning("No utterances found in database")
                return RegenerationReport(
                    total_utterances=0,
                    processed=0,
                    failed=0,
                    entities_created=0,
                    relationships_created=0,
                    duration_seconds=0.0,
                    errors=[],
                )

            # Step 2: Delete entity graph
            logger.info("Step 2: Deleting entity graph...")
            self._delete_graph_entities()
            logger.info("Entity graph deleted (utterances preserved)")

            # Step 3: Re-extract
            logger.info("Step 3: Re-extracting entities...")
            total_entities = 0
            total_relationships = 0
            processed = 0
            failed = 0
            errors = []

            for i, utterance in enumerate(utterances, 1):
                try:
                    entities, relationships = await self._extract_and_store(utterance)
                    total_entities += entities
                    total_relationships += relationships
                    processed += 1

                    # Progress logging
                    if i % 10 == 0 or i == len(utterances):
                        logger.info(
                            f"Progress: {i}/{len(utterances)} "
                            f"({(i/len(utterances)*100):.1f}%) - "
                            f"{total_entities} entities, {total_relationships} relationships"
                        )

                except Exception as e:
                    failed += 1
                    error_msg = f"Utterance {utterance.utterance_id[:8]}: {str(e)}"
                    errors.append(error_msg)
                    logger.error(f"Failed to process utterance: {error_msg}")
                    # Continue with next utterance

            # Step 4: Report
            duration = (datetime.now() - start_time).total_seconds()

            report = RegenerationReport(
                total_utterances=len(utterances),
                processed=processed,
                failed=failed,
                entities_created=total_entities,
                relationships_created=total_relationships,
                duration_seconds=duration,
                errors=errors,
            )

            logger.info("=" * 60)
            logger.info("Regeneration complete!")
            logger.info(f"Processed: {processed}/{len(utterances)} utterances")
            logger.info(f"Entities: {total_entities}")
            logger.info(f"Relationships: {total_relationships}")
            logger.info(f"Failed: {failed}")
            logger.info(f"Duration: {duration:.2f}s")
            logger.info("=" * 60)

            return report

        except Exception as e:
            logger.error(f"Regeneration failed: {e}")
            raise

    async def regenerate_conversation(self, conversation_id: str) -> RegenerationReport:
        """Regenerate graph for a specific conversation.

        Useful for:
        - Testing on subset of data
        - Incremental regeneration
        - Development and debugging

        Args:
            conversation_id: ID of conversation to regenerate

        Returns:
            RegenerationReport with statistics
        """
        start_time = datetime.now()

        logger.info(f"Regenerating conversation: {conversation_id}")

        try:
            # Fetch utterances for this conversation
            utterances = self._fetch_conversation_utterances(conversation_id)
            logger.info(f"Found {len(utterances)} utterances in conversation")

            if len(utterances) == 0:
                logger.warning(f"No utterances found for conversation {conversation_id}")
                return RegenerationReport(
                    total_utterances=0,
                    processed=0,
                    failed=0,
                    entities_created=0,
                    relationships_created=0,
                    duration_seconds=0.0,
                    errors=[],
                )

            # Delete entities for this conversation only
            self._delete_conversation_entities(conversation_id)

            # Re-extract
            total_entities = 0
            total_relationships = 0
            processed = 0
            failed = 0
            errors = []

            for utterance in utterances:
                try:
                    entities, relationships = await self._extract_and_store(utterance)
                    total_entities += entities
                    total_relationships += relationships
                    processed += 1
                except Exception as e:
                    failed += 1
                    errors.append(f"{utterance.utterance_id[:8]}: {str(e)}")
                    logger.error(f"Failed utterance: {e}")

            duration = (datetime.now() - start_time).total_seconds()

            report = RegenerationReport(
                total_utterances=len(utterances),
                processed=processed,
                failed=failed,
                entities_created=total_entities,
                relationships_created=total_relationships,
                duration_seconds=duration,
                errors=errors,
            )

            logger.info(
                f"Conversation regeneration complete: {processed}/{len(utterances)} processed"
            )

            return report

        except Exception as e:
            logger.error(f"Conversation regeneration failed: {e}")
            raise

    def _fetch_all_utterances(self) -> list[Utterance]:
        """Fetch all utterances from Neo4j.

        Returns utterances in chronological order to maintain conversation context.

        Returns:
            List of Utterance objects ordered by timestamp
        """
        query = """
        MATCH (u:Utterance)
        OPTIONAL MATCH (u)-[:PART_OF]->(c:ConversationEvent)
        RETURN
            u.utterance_id AS utterance_id,
            u.text AS text,
            u.timestamp AS timestamp,
            u.metadata AS metadata,
            c.conversation_id AS conversation_id
        ORDER BY u.timestamp ASC
        """

        self.connection.connect()
        results = self.connection.execute_query(query)
        self.connection.disconnect()

        utterances = []
        for r in results:
            utterances.append(
                Utterance(
                    utterance_id=r["utterance_id"],
                    conversation_id=r.get("conversation_id", "unknown"),
                    text=r["text"],
                    timestamp=(
                        r["timestamp"].to_native()
                        if hasattr(r["timestamp"], "to_native")
                        else r["timestamp"]
                    ),
                    metadata=r.get("metadata"),
                )
            )

        return utterances

    def _fetch_conversation_utterances(self, conversation_id: str) -> list[Utterance]:
        """Fetch utterances for a specific conversation.

        Args:
            conversation_id: Conversation ID

        Returns:
            List of Utterance objects for that conversation
        """
        query = """
        MATCH (c:ConversationEvent {conversation_id: $conversation_id})
              <-[:PART_OF]-(u:Utterance)
        RETURN
            u.utterance_id AS utterance_id,
            u.text AS text,
            u.timestamp AS timestamp,
            u.metadata AS metadata,
            c.conversation_id AS conversation_id
        ORDER BY u.timestamp ASC
        """

        self.connection.connect()
        results = self.connection.execute_query(query, {"conversation_id": conversation_id})
        self.connection.disconnect()

        utterances = []
        for r in results:
            utterances.append(
                Utterance(
                    utterance_id=r["utterance_id"],
                    conversation_id=r["conversation_id"],
                    text=r["text"],
                    timestamp=(
                        r["timestamp"].to_native()
                        if hasattr(r["timestamp"], "to_native")
                        else r["timestamp"]
                    ),
                    metadata=r.get("metadata"),
                )
            )

        return utterances

    def _delete_graph_entities(self):
        """Delete all :__Entity__ nodes and their relationships.

        Targets only the user-facing entity graph (nodes carrying the
        :__Entity__ base label).  Provenance nodes (:__Provenance__:*)
        are NOT deleted -- they are an independent audit trail and are
        preserved across regenerations.

        This removes the materialized entity-graph view but preserves the
        source of truth (ConversationEvent, Utterance, and all
        :__Provenance__ nodes).

        DETACH DELETE removes nodes AND all their relationships automatically.
        """
        logger.info("Deleting all entity nodes...")

        query = """
        MATCH (e:__Entity__)
        DETACH DELETE e
        """

        self.connection.connect()
        self.connection.execute_write(query)

        # Verify deletion
        count_query = "MATCH (e:__Entity__) RETURN count(e) AS count"
        result = self.connection.execute_query(count_query)
        entity_count = result[0]["count"] if result else 0

        self.connection.disconnect()

        if entity_count > 0:
            raise RuntimeError(f"Failed to delete all entities: {entity_count} remain")

        logger.info("All entity nodes deleted successfully")

    def _delete_conversation_entities(self, conversation_id: str):
        """Delete entities for a specific conversation.

        Args:
            conversation_id: Conversation ID
        """
        logger.info(f"Deleting entities for conversation {conversation_id}...")

        query = """
        MATCH (c:ConversationEvent {conversation_id: $conversation_id})
              <-[:PART_OF]-(u:Utterance)
              -[:HAS_ENTITY]->(e:__Entity__)
        DETACH DELETE e
        """

        self.connection.connect()
        self.connection.execute_write(query, {"conversation_id": conversation_id})
        self.connection.disconnect()

        logger.info("Conversation entities deleted")

    async def _extract_and_store(self, utterance: Utterance) -> tuple[int, int]:
        """Re-extract entities from utterance via ExtractionPipeline.

        Args:
            utterance: Utterance to process.

        Returns:
            Tuple of (entities_count, relationships_count).
        """
        try:
            result = await self._pipeline.extract_from_utterance(
                utterance=utterance.text,
                conversation_history=[],
                event_id=f"regen_{utterance.utterance_id}",
                session_id=utterance.conversation_id or "regeneration",
            )

            if result.entities:
                self.graph_store.store_validated_entities(
                    entities=result.entities,
                    relationships=result.relationships,
                    utterance_id=utterance.utterance_id,
                    ontology_version=self.config.ontology_version,
                )
                return len(result.entities), len(result.relationships)

            return 0, 0

        except Exception as e:
            logger.error("Extraction failed for utterance %s: %s", utterance.utterance_id, e)
            raise
