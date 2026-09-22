"""Initialize Neo4j Schema.

Creates indexes and constraints for the knowledge graph,
including the vector index for semantic search.

Usage:
    python scripts/initialize_schema.py
"""

import logging
import sys
from pathlib import Path

# Running this script by path puts scripts/ on sys.path[0], NOT the repo
# root -- so `import backend` would fail. Add the repo root first (mirrors
# scripts/mist_admin.py:81-83).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.factories import build_graph_store  # noqa: E402
from backend.knowledge.config import get_config  # noqa: E402

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


def main():
    """Initialize Neo4j schema."""
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

        # Initialize graph store. build_graph_store wires the real
        # connection and embedding generator the same way the production
        # backend does (backend/factories.py:176-187). The embedding
        # generator it constructs is lazy -- EmbeddingGenerator.__init__
        # never loads the model (backend/knowledge/embeddings/
        # embedding_generator.py:30-41) -- and initialize_schema() below
        # never calls generate_embedding/generate_embeddings, so this does
        # no network I/O even on a cold model cache.
        graph_store = build_graph_store(config)
        logger.info("GraphStore initialized")

        # Initialize schema
        print("Initializing schema...")
        graph_store.initialize_schema()

        print("\n" + "=" * 60)
        print("SCHEMA INITIALIZATION COMPLETE")
        print("=" * 60)
        print("\nNext steps:")
        print("  1. Run: python -m scripts.mist_admin seed")
        print("     (Restore the seed baseline: entities + relationships + embeddings)")
        print("  2. Optional: python -m scripts.mist_admin vault-rebuild --confirm")
        print("     (Drop and re-index the vault sidecar from the vault corpus on disk --")
        print("     read-path only; this writes no graph content. For a graph rebuild,")
        print("     see: python -m scripts.mist_admin graph-rebuild-from-log --dry-run)")
        print()

    except Exception as e:
        logger.error(f"Schema initialization failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
