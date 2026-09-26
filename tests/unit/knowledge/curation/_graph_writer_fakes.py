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


# The EXTRACTED_FROM edge MERGE in `_create_provenance_edge`. Tests that look
# for the edge write match on this, not on the bare relationship type: the
# entity upsert's replay guard also names EXTRACTED_FROM (it reads the edge).
PROVENANCE_EDGE_MERGE = "MERGE (e)-[r:EXTRACTED_FROM]->(ctx)"

# The replay guard's text in `_upsert_entity`, as `ReplayGraphConnection`
# recognises it. All three fragments must be present for the fake to apply the
# guard; a guard reworded in production reads as ABSENT here, so the replay
# tests go red (fail closed) rather than passing without a guard.
UPSERT_REPLAY_GUARD_FRAGMENTS = (
    "OPTIONAL MATCH (:__Entity__ {id: $entity_id})-[seen:EXTRACTED_FROM]->"
    "(:ConversationContext {conversation_id: $session_id}) ",
    "WHERE seen.source_utterance_id = $event_id ",
    "ON MATCH SET e.confidence = CASE WHEN event_already_applied THEN e.confidence ",
)

_CONTEXT_MERGE = "MERGE (ctx:__Provenance__:ConversationContext"
_ENTITY_MERGE = "MERGE (e:__Entity__ {id: $entity_id})"


class SimulatedCrashError(RuntimeError):
    """Stands in for a process kill at a chosen statement."""


class ReplayGraphConnection(FakeNeo4jConnection):
    """A stateful fake that evaluates the writer's confidence-relevant Cypher.

    Models exactly what the entity-confidence replay guard depends on, keyed on
    the statements `CurationGraphWriter.write` issues on the conversational
    path:

    - the ConversationContext MERGE (records the session);
    - the entity upsert: ON CREATE sets `confidence = $confidence`; ON MATCH
      sets `confidence = max(confidence, $reinforced)`, unless the guard text
      (`UPSERT_REPLAY_GUARD_FRAGMENTS`) is present AND an EXTRACTED_FROM edge
      from this entity to `$session_id`'s context has
      `source_utterance_id == $event_id`;
    - the EXTRACTED_FROM MERGE: MATCH entity, MATCH context, then set
      `source_utterance_id = $event_id` on both branches (last-writer-wins).

    Every other write (LearningEvent) is recorded and otherwise ignored. It does
    NOT evaluate Cypher: whether Neo4j itself evaluates the guard as intended is
    what `tests/integration/extraction_backlog/test_crash_replay_canonical.py`
    checks against the eval instance.

    `crash_on_write`: 1-based index of a write that raises `SimulatedCrashError`
    BEFORE it takes effect (a kill between two statements); None never crashes.
    """

    def __init__(self, *, crash_on_write: int | None = None) -> None:
        super().__init__()
        self.contexts: set[str] = set()
        self.node_confidence: dict[str, float] = {}
        # (entity_id, session_id) -> source_utterance_id
        self.extracted_from: dict[tuple[str, str], str] = {}
        self.crash_on_write = crash_on_write

    def execute_write(self, query, params=None):
        super().execute_write(query, params)
        if self.crash_on_write is not None and len(self.writes) == self.crash_on_write:
            self.crash_on_write = None
            raise SimulatedCrashError(f"killed before write {len(self.writes)}")
        p = params or {}
        if _CONTEXT_MERGE in query:
            self.contexts.add(p["session_id"])
        elif _ENTITY_MERGE in query:
            self._upsert(query, p)
        elif PROVENANCE_EDGE_MERGE in query:
            key = (p["entity_id"], p["session_id"])
            if p["entity_id"] in self.node_confidence and p["session_id"] in self.contexts:
                self.extracted_from[key] = p["event_id"]
        return []

    def _upsert(self, query: str, p: dict[str, Any]) -> None:
        entity_id = p["entity_id"]
        if entity_id not in self.node_confidence:
            self.node_confidence[entity_id] = p["confidence"]
            return
        if all(fragment in query for fragment in UPSERT_REPLAY_GUARD_FRAGMENTS):
            # Indexing, not .get(): a guard whose parameters are not bound is a
            # ParameterMissing error in Neo4j, and must not read as "seen".
            recorded = self.extracted_from.get((entity_id, p["session_id"]))
            if recorded is not None and recorded == p["event_id"]:
                return
        self.node_confidence[entity_id] = max(self.node_confidence[entity_id], p["reinforced"])


def make_replay_writer(
    conn: ReplayGraphConnection,
) -> CurationGraphWriter:
    """Build a CurationGraphWriter over a (possibly shared) stateful fake graph."""
    return CurationGraphWriter(
        executor=FakeGraphExecutor(connection=conn),
        embedding_provider=FakeEmbeddingGenerator(),
        confidence_manager=ConfidenceManager(),
        rebuild_stamps=TEST_REBUILD_STAMPS,
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
