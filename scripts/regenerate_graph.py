"""Graph Regeneration Script - Complete All-In-One Solution.

Handles everything needed for graph regeneration:
1. Checks Neo4j connection
2. Initializes schema (indexes, constraints, vector index)
3. Rebuilds knowledge graph from immutable utterances
4. Generates embeddings for all entities

This script demonstrates the event sourcing architecture:
- Utterances are the immutable source of truth
- Knowledge graph is a materialized view
- Can always rebuild from scratch

Usage:
    python regenerate_graph.py

Options:
    --conversation-id <id>  : Regenerate specific conversation only
    --dry-run              : Show what would be done without executing

Example:
    # Regenerate entire graph (includes schema initialization)
    python regenerate_graph.py

    # Regenerate specific conversation
    python regenerate_graph.py --conversation-id user-123-session-456

    # Dry run (preview)
    python regenerate_graph.py --dry-run

Note: This script automatically handles schema initialization, so you don't
need to run any other scripts first. Just make sure Neo4j is running!
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from typing import TYPE_CHECKING

from backend.factories import build_graph_regenerator
from backend.knowledge.config import get_config
from backend.knowledge.storage import GraphStore, Neo4jConnection

if TYPE_CHECKING:
    from backend.knowledge.regeneration.graph_regenerator import GraphRegenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


async def regenerate_all(regenerator: GraphRegenerator, dry_run: bool = False):
    """Regenerate entire knowledge graph.

    Args:
        regenerator: GraphRegenerator instance
        dry_run: If True, show what would be done without executing
    """
    if dry_run:
        logger.info("DRY RUN MODE - No changes will be made")
        logger.info("Would regenerate entire knowledge graph from all utterances")
        return

    logger.info("Starting full graph regeneration...")
    report = await regenerator.regenerate_all()

    print("\n" + "=" * 60)
    print("REGENERATION COMPLETE")
    print("=" * 60)
    print(f"Total Utterances:     {report.total_utterances}")
    print(f"Successfully Processed: {report.processed}")
    print(f"Failed:               {report.failed}")
    print(f"Entities Created:     {report.entities_created}")
    print(f"Relationships Created: {report.relationships_created}")
    print(f"Duration:             {report.duration_seconds:.2f}s")

    if report.errors:
        print(f"\nErrors ({len(report.errors)}):")
        for error in report.errors[:10]:  # Show first 10 errors
            print(f"  - {error}")
        if len(report.errors) > 10:
            print(f"  ... and {len(report.errors) - 10} more")

    print("=" * 60)

    # Helpful next steps
    if report.processed > 0:
        print("\nNext steps:")
        print("  1. Test vector search:")
        print("     python test_vector_search.py")
        print()
        print("  2. Test knowledge retrieval:")
        print("     python test_knowledge_retrieval.py")
        print()
        print("  3. View graph in Neo4j Browser:")
        print("     http://localhost:7474")
        print()

    return report


async def regenerate_conversation(
    regenerator: GraphRegenerator, conversation_id: str, dry_run: bool = False
):
    """Regenerate specific conversation.

    Args:
        regenerator: GraphRegenerator instance
        conversation_id: ID of conversation to regenerate
        dry_run: If True, show what would be done without executing
    """
    if dry_run:
        logger.info("DRY RUN MODE - No changes will be made")
        logger.info(f"Would regenerate conversation: {conversation_id}")
        return

    logger.info(f"Regenerating conversation: {conversation_id}")
    report = await regenerator.regenerate_conversation(conversation_id)

    print("\n" + "=" * 60)
    print(f"CONVERSATION REGENERATION COMPLETE: {conversation_id}")
    print("=" * 60)
    print(f"Total Utterances:     {report.total_utterances}")
    print(f"Successfully Processed: {report.processed}")
    print(f"Failed:               {report.failed}")
    print(f"Entities Created:     {report.entities_created}")
    print(f"Relationships Created: {report.relationships_created}")
    print(f"Duration:             {report.duration_seconds:.2f}s")

    if report.errors:
        print(f"\nErrors ({len(report.errors)}):")
        for error in report.errors:
            print(f"  - {error}")

    print("=" * 60)

    return report


async def initialize_schema(config):
    """Initialize Neo4j schema (indexes and constraints).

    Creates all necessary indexes including vector index for embeddings.

    Args:
        config: Knowledge configuration

    Returns:
        True if successful, False otherwise
    """
    logger.info("Initializing Neo4j schema...")

    try:
        graph_store = GraphStore(config)
        graph_store.initialize_schema()
        logger.info("[OK] Schema initialized successfully")
        return True

    except Exception as e:
        logger.error(f"Schema initialization failed: {e}")
        return False


async def check_neo4j_connection(config):
    """Verify Neo4j connection before regeneration.

    Args:
        config: Knowledge configuration

    Returns:
        True if connection successful, False otherwise
    """
    logger.info("Checking Neo4j connection...")

    try:
        connection = Neo4jConnection(config.neo4j)
        connection.connect()

        # Test query
        result = connection.execute_query("RETURN 1 AS test")
        if result and result[0].get("test") == 1:
            logger.info("[OK] Neo4j connection successful")
            connection.disconnect()
            return True
        else:
            logger.error("Neo4j connection test failed")
            connection.disconnect()
            return False

    except Exception as e:
        logger.error(f"Neo4j connection failed: {e}")
        return False


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Regenerate knowledge graph from utterances",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Regenerate entire graph
  python regenerate_graph.py

  # Regenerate specific conversation
  python regenerate_graph.py --conversation-id user-123-session-456

  # Dry run (preview)
  python regenerate_graph.py --dry-run
        """,
    )

    parser.add_argument("--conversation-id", type=str, help="Regenerate specific conversation only")

    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be done without executing"
    )

    args = parser.parse_args()

    # Load configuration
    try:
        config = get_config()
        logger.info("Configuration loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)

    # Check Neo4j connection
    if not await check_neo4j_connection(config):
        logger.error("Cannot proceed without Neo4j connection")
        sys.exit(1)

    # Initialize schema (indexes, constraints, vector index)
    logger.info("Step 1: Initializing schema...")
    if not await initialize_schema(config):
        logger.error("Failed to initialize schema")
        sys.exit(1)

    # Initialize regenerator
    logger.info("Step 2: Initializing regenerator...")
    try:
        regenerator = build_graph_regenerator(config)
        logger.info("GraphRegenerator initialized")
    except Exception as e:
        logger.error(f"Failed to initialize GraphRegenerator: {e}")
        sys.exit(1)

    # Execute regeneration
    try:
        if args.conversation_id:
            # Regenerate specific conversation
            await regenerate_conversation(regenerator, args.conversation_id, dry_run=args.dry_run)
        else:
            # Regenerate entire graph
            await regenerate_all(regenerator, dry_run=args.dry_run)

        logger.info("Regeneration completed successfully")

    except KeyboardInterrupt:
        logger.warning("Regeneration interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Regeneration failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
