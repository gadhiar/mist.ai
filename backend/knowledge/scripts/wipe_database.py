"""
Wipe Neo4j Database Completely

DANGER: This script deletes EVERYTHING from Neo4j.
- All nodes (ConversationEvents, Utterances, Entities)
- All relationships
- All indexes and constraints

Use this to start completely fresh.
"""

import logging
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from backend.knowledge.config import KnowledgeConfig
from backend.knowledge.storage.neo4j_connection import Neo4jConnection

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def wipe_database():
    """Completely wipe the Neo4j database"""

    print("\n" + "=" * 60)
    print("WARNING: DATABASE WIPE")
    print("=" * 60)
    print("This will DELETE EVERYTHING from Neo4j:")
    print("  - All nodes (ConversationEvents, Utterances, Entities)")
    print("  - All relationships")
    print("  - All indexes and constraints")
    print("=" * 60)

    response = input("\nType 'DELETE EVERYTHING' to confirm: ")

    if response != "DELETE EVERYTHING":
        print("\nOperation cancelled. Database not modified.")
        return

    print("\nConnecting to Neo4j...")
    config = KnowledgeConfig.from_env()
    connection = Neo4jConnection(config.neo4j)
    connection.connect()

    print("\n[1/3] Deleting all nodes and relationships...")
    delete_query = """
    MATCH (n)
    DETACH DELETE n
    """
    connection.execute_write(delete_query)
    logger.info("All nodes and relationships deleted")

    print("[2/3] Dropping all constraints...")
    # Get all constraints
    constraints_query = "SHOW CONSTRAINTS"
    try:
        constraints = connection.execute_query(constraints_query)
        for constraint in constraints:
            constraint_name = constraint.get("name")
            if constraint_name:
                try:
                    drop_query = f"DROP CONSTRAINT {constraint_name} IF EXISTS"
                    connection.execute_write(drop_query)
                    logger.info(f"Dropped constraint: {constraint_name}")
                except Exception as e:
                    logger.warning(f"Could not drop constraint {constraint_name}: {e}")
    except Exception as e:
        logger.warning(f"Could not list constraints: {e}")

    print("[3/3] Dropping all indexes...")
    # Get all indexes
    indexes_query = "SHOW INDEXES"
    try:
        indexes = connection.execute_query(indexes_query)
        for index in indexes:
            index_name = index.get("name")
            # Skip constraint-backed indexes (they're dropped with constraints)
            if index_name and index.get("type") != "CONSTRAINT":
                try:
                    drop_query = f"DROP INDEX {index_name} IF EXISTS"
                    connection.execute_write(drop_query)
                    logger.info(f"Dropped index: {index_name}")
                except Exception as e:
                    logger.warning(f"Could not drop index {index_name}: {e}")
    except Exception as e:
        logger.warning(f"Could not list indexes: {e}")

    connection.disconnect()

    print("\n" + "=" * 60)
    print("DATABASE WIPED SUCCESSFULLY")
    print("=" * 60)
    print("The Neo4j database is now completely empty.")
    print("Run seed_from_docs.py to populate with MIST documentation.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    wipe_database()
