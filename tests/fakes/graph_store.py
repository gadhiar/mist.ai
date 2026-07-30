"""FakeGraphStore -- in-memory test double for GraphStore.

`current_ontology_version` is the only surface GraphRegenerator still needs
(typed structurally via `_OntologyVersionSource` in
backend/knowledge/curation/graph_regenerator.py).

R1.3 deleted the real `GraphStore.upsert_user` (Task 3) and
`GraphStore.mark_orphaned_by_provenance_path` / `get_orphaned_provenance_paths`
(Task 5). This fake's mirrors of those three methods are trimmed too, in
this task: a fake mirroring methods that exist on no real class is a trap
for the next reader, and their last method-level consumers (this file's own
prior test coverage, plus test_knowledge_retriever_orphan_filter.py's
FakeGraphStore-only lifecycle simulation) were deleted alongside them.

`mark_orphaned_calls` and `upsert_user_calls` stay as plain instance
attributes -- always `[]`, since nothing populates them anymore --
because test_graph_regenerator.py still asserts `== []` on both as a
call-trap proving GraphRegenerator's no-op `rebuild_from_path` never
reaches the graph store. That file is the last consumer of GraphRegenerator
itself and is deleted wholesale in Task 6, which is where these two
attributes should go too.

Also exposes assertion helpers for test readability:
  - add_triple / get_triple / count_traits / has_trait
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FakeTriple:
    """Minimal triple record tracked by FakeGraphStore."""

    subject: str
    predicate: str
    object: str
    derived_from_path: str = ""
    status: str = "active"


class FakeGraphStore:
    """In-memory test double for GraphStore.

    Tracks typed triples via `add_triple` / `get_triple` for tests that
    arrange graph state directly (no upsert path exists on the fake
    anymore; the real GraphStore.upsert_user it used to mirror is retired).
    """

    _ONTOLOGY_VERSION = "1.1.0"

    def __init__(self) -> None:
        self._triples: list[FakeTriple] = []
        self.mark_orphaned_calls: list[str] = []
        self.upsert_user_calls: list[dict] = []

    def current_ontology_version(self) -> str:
        return self._ONTOLOGY_VERSION

    # ------------------------------------------------------------------
    # Test helpers
    # ------------------------------------------------------------------

    def add_triple(
        self,
        subject: str,
        predicate: str,
        object: str,
        derived_from_path: str = "",
        status: str = "active",
    ) -> None:
        """Pre-seed a triple for tests that arrange graph state directly."""
        self._triples.append(
            FakeTriple(
                subject=subject,
                predicate=predicate,
                object=object,
                derived_from_path=derived_from_path,
                status=status,
            )
        )

    def get_triple(self, subject: str, predicate: str, object: str) -> FakeTriple | None:
        """Retrieve a triple by (subject, predicate, object) key."""
        for triple in self._triples:
            if (
                triple.subject == subject
                and triple.predicate == predicate
                and triple.object == object
            ):
                return triple
        return None

    def has_trait(self, trait_slug: str) -> bool:
        """Return True if a HAS_TRAIT triple exists for the given slug."""
        return self.get_triple("mist-identity", "HAS_TRAIT", trait_slug) is not None

    def count_traits(self) -> int:
        """Return the number of HAS_TRAIT triples (deduped)."""
        return sum(
            1 for t in self._triples if t.predicate == "HAS_TRAIT" and t.subject == "mist-identity"
        )
