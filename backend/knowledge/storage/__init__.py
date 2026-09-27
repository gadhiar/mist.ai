"""Knowledge storage module.

PEP 562 lazy exports: importing this package (or any of its submodules,
which Python resolves by importing the package first) must not pull in
`neo4j` -- and transitively `pandas`/`pyarrow` via neo4j's own optional-deps
probe -- or `lancedb` just to read one name off the package. Every name in
`__all__` stays importable via `from backend.knowledge.storage import X`;
only the eager `import neo4j` / `import lancedb` at package-init time is
removed.
"""

__all__ = ["Neo4jConnection", "GraphStore", "LanceDBVectorStore"]


def __getattr__(name: str):
    """Lazy import of this package's public names.

    Deferring `GraphStore`/`Neo4jConnection` (not just the pre-existing
    `LanceDBVectorStore`) keeps `neo4j` -- and pandas/pyarrow behind it --
    out of any import that only needs another submodule of this package
    (e.g. `backend.knowledge.storage.graph_executor`), since importing a
    submodule always runs the package `__init__` first.
    """
    if name == "GraphStore":
        from backend.knowledge.storage.graph_store import GraphStore

        return GraphStore
    if name == "Neo4jConnection":
        from backend.knowledge.storage.neo4j_connection import Neo4jConnection

        return Neo4jConnection
    if name == "LanceDBVectorStore":
        from backend.knowledge.storage.vector_store import LanceDBVectorStore

        return LanceDBVectorStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals()))
