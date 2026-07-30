"""Shared fakes for curation graph-writer unit tests.

Built on the canonical test doubles in `tests/mocks/` rather than a private
copy -- `tests/CLAUDE.md`'s Mocking Rules require mocking only at I/O
boundaries and reusing the doubles in `tests/mocks/`. `FakeNeo4jConnection`
already records every write as `(query, params)`, so `make_writer` returns
the connection (not the `FakeGraphExecutor` wrapper) for assertions.
`ConfidenceManager` is pure computation, not an I/O boundary, so tests use
the real one rather than faking it.
"""

from typing import Any

from backend.knowledge.curation.confidence import ConfidenceManager
from backend.knowledge.curation.graph_writer import CurationGraphWriter, RebuildStamps
from tests.mocks.embeddings import FakeEmbeddingGenerator
from tests.mocks.neo4j import FakeGraphExecutor, FakeNeo4jConnection


def make_writer(
    rebuild_stamps: RebuildStamps | None = None,
) -> tuple[CurationGraphWriter, FakeNeo4jConnection]:
    """Build a CurationGraphWriter over the canonical recording connection."""
    conn = FakeNeo4jConnection()
    writer = CurationGraphWriter(
        executor=FakeGraphExecutor(connection=conn),
        embedding_provider=FakeEmbeddingGenerator(),
        confidence_manager=ConfidenceManager(),
        rebuild_stamps=rebuild_stamps,
    )
    return writer, conn


def writes_matching(conn: FakeNeo4jConnection, needle: str) -> list[tuple[str, dict[str, Any]]]:
    """Return recorded writes whose query contains `needle`."""
    return [(q, p or {}) for q, p in conn.writes if needle in q]
