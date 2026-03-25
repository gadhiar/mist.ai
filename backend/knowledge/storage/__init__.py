"""Knowledge storage module."""

from backend.knowledge.storage.graph_store import GraphStore
from backend.knowledge.storage.neo4j_connection import Neo4jConnection

__all__ = ["Neo4jConnection", "GraphStore", "LanceDBVectorStore"]


def __getattr__(name: str):
    """Lazy import for LanceDBVectorStore to avoid requiring lancedb at import time."""
    if name == "LanceDBVectorStore":
        from backend.knowledge.storage.vector_store import LanceDBVectorStore

        return LanceDBVectorStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
