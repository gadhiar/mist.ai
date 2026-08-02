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

# `rebuild_stamps` is a required dependency on CurationGraphWriter, so tests
# that do not care about stamp VALUES still have to supply one. These are the
# canonical test stamps -- deliberately not the production values, so a test
# asserting on them cannot accidentally pass by matching a real default.
TEST_REBUILD_STAMPS = RebuildStamps(
    ontology_version="1.4.0-test",
    extraction_version="2026-06-14-r5-test",
    model_hash="test-model-hash",
)


def make_writer(
    rebuild_stamps: RebuildStamps | None = None,
) -> tuple[CurationGraphWriter, FakeNeo4jConnection]:
    """Build a CurationGraphWriter over the canonical recording connection."""
    conn = FakeNeo4jConnection()
    writer = CurationGraphWriter(
        executor=FakeGraphExecutor(connection=conn),
        embedding_provider=FakeEmbeddingGenerator(),
        confidence_manager=ConfidenceManager(),
        rebuild_stamps=rebuild_stamps or TEST_REBUILD_STAMPS,
    )
    return writer, conn


def writes_matching(conn: FakeNeo4jConnection, needle: str) -> list[tuple[str, dict[str, Any]]]:
    """Return recorded writes whose query contains `needle`."""
    return [(q, p or {}) for q, p in conn.writes if needle in q]
