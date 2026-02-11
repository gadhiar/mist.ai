"""
Initialize Neo4j Schema

Creates indexes and constraints for the knowledge graph,
including the vector index for semantic search.

Usage:
    python initialize_schema.py
"""

import logging
import sys

from backend.knowledge.config import get_config
from backend.knowledge.storage import GraphStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


def main():
    """Initialize Neo4j schema"""
    print("=" * 60)
    print("Neo4j Schema Initialization")
    print("=" * 60)
    print("\nThis will create:")
    print("  - Uniqueness constraints")
    print("  - Performance indexes")
    print("  - Vector index for semantic search")
    print()

    try:
        # Load configuration
        config = get_config()
        logger.info("Configuration loaded")

        # Initialize graph store
        graph_store = GraphStore(config)
        logger.info("GraphStore initialized")

        # Initialize schema
        print("Initializing schema...")
        graph_store.initialize_schema()

        print("\n" + "=" * 60)
        print("SCHEMA INITIALIZATION COMPLETE")
        print("=" * 60)
        print("\nNext steps:")
        print("  1. Run: python regenerate_graph.py")
        print("     (This will add embeddings to all entities)")
        print("  2. Run: python test_vector_search.py")
        print("     (Test semantic search)")
        print()

    except Exception as e:
        logger.error(f"Schema initialization failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
