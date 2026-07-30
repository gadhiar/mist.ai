"""FakeGraphStore — in-memory test double for GraphStoreProtocol.

Satisfies the GraphStoreProtocol surface required by GraphRegenerator:
  - mark_orphaned_by_provenance_path
  - current_ontology_version

`upsert_user` is retained as a transitional call-trap after R1.3 deleted
the real `GraphStore.upsert_user` and its `GraphStoreProtocol` declaration:
tests assert `upsert_user_calls == []` to prove the retired Bucket-1 write
path never fires. R1.3 Task 6 removes it once the last consumer is gone.

Also exposes assertion helpers for test readability:
  - add_triple / get_triple / count_traits / has_trait

Schema alignment (Phase 5.5 Bucket 1 fix):
  The real mark_orphaned_by_provenance_path queries DERIVED_FROM relationship-
  type edges pointing at :__Provenance__:VaultNote nodes. This fake mirrors
  that schema: upsert_user writes FakeDerivedFromEdge records, and
  mark_orphaned_by_provenance_path marks those edges by path. This ensures
  the fake catches the same class of bug as the real Neo4j implementation
  rather than masking it via property-based triple matching.
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


@dataclass
class FakeDerivedFromEdge:
    """Provenance edge: typed entity -> VaultNote (mirrors DERIVED_FROM schema).

    Keyed by (entity_id, path). status mirrors the real Neo4j edge status
    so mark_orphaned_by_provenance_path can find edges by path.
    """

    entity_id: str
    path: str
    status: str = "active"


class FakeGraphStore:
    """In-memory test double for GraphStoreProtocol.

    Tracks typed triples and DERIVED_FROM provenance edges separately.
    Idempotent upsert: calling upsert_user with the same display_name twice
    writes only one triple and one provenance edge (dedup).

    mark_orphaned_by_provenance_path mirrors the real Neo4j implementation:
    it finds DERIVED_FROM edges by path and marks their status='orphaned'.
    This catches the bug class where upsert writes typed edges but omits the
    DERIVED_FROM provenance edge, leaving mark_orphaned with nothing to find.

    Supports `get_orphaned_provenance_paths` for retry_orphaned tests:
    returns the distinct set of paths for which DERIVED_FROM edges are orphaned.
    """

    _ONTOLOGY_VERSION = "1.1.0"

    def __init__(self) -> None:
        self._triples: list[FakeTriple] = []
        # DERIVED_FROM provenance edges (entity_id, path) -> FakeDerivedFromEdge
        self._provenance_edges: dict[tuple[str, str], FakeDerivedFromEdge] = {}
        self.mark_orphaned_calls: list[str] = []
        self.upsert_user_calls: list[dict] = []

    # ------------------------------------------------------------------
    # GraphStoreProtocol methods
    # ------------------------------------------------------------------

    async def mark_orphaned_by_provenance_path(self, path: str) -> int:
        """Mark all DERIVED_FROM edges pointing at path as 'orphaned'.

        Mirrors the real Cypher: MATCH ()-[d:DERIVED_FROM]->(vn:VaultNote {path})
        WHERE d.status <> 'orphaned' SET d.status = 'orphaned'.
        Returns count of edges marked.

        Also propagates orphaned status to any FakeTriple records sharing the
        same derived_from_path so that triple-level assertions in existing tests
        (triple.status == 'orphaned') continue to hold after the schema migration
        from property-based to relationship-based provenance tracking.
        """
        self.mark_orphaned_calls.append(path)
        count = 0
        for edge in self._provenance_edges.values():
            if edge.path == path and edge.status != "orphaned":
                edge.status = "orphaned"
                count += 1
        # Propagate to _triples so triple-level assertions remain valid.
        for triple in self._triples:
            if triple.derived_from_path == path and triple.status != "orphaned":
                triple.status = "orphaned"
        return count

    def current_ontology_version(self) -> str:
        return self._ONTOLOGY_VERSION

    async def get_orphaned_provenance_paths(self) -> list[str]:
        """Return distinct paths for which DERIVED_FROM edges are orphaned.

        Mirrors real Cypher: MATCH ()-[d:DERIVED_FROM]->(vn:VaultNote)
        WHERE d.status = 'orphaned' RETURN DISTINCT vn.path.
        """
        seen: set[str] = set()
        result: list[str] = []
        for edge in self._provenance_edges.values():
            if edge.status == "orphaned" and edge.path not in seen:
                seen.add(edge.path)
                result.append(edge.path)
        return result

    async def upsert_user(self, parsed_user, derived_from_path: str) -> int:
        """Write ParsedUser edge targets as graph triples (idempotent).

        Mirrors the real upsert_user: for each typed entity, writes a typed
        triple AND a DERIVED_FROM provenance edge to the VaultNote at
        derived_from_path. mark_orphaned_by_provenance_path finds edges by path.
        """
        self.upsert_user_calls.append(
            {"parsed_user": parsed_user, "derived_from_path": derived_from_path}
        )
        written = 0
        user_id = parsed_user.user_id
        section_map = {
            "USES": parsed_user.tools_and_technologies,
            "EXPERT_IN": parsed_user.expertise,
            "LEARNING": parsed_user.currently_learning,
            "WORKS_ON": parsed_user.projects,
            "WORKS_AT": parsed_user.affiliations,
            "INTERESTED_IN": parsed_user.interests,
            "HAS_GOAL": parsed_user.goals,
            "PREFERS": parsed_user.preferences,
            "KNOWS_PERSON": parsed_user.people,
        }
        for predicate, targets in section_map.items():
            for target in targets:
                target_id = f"entity-{target.lower().replace(' ', '-')}"
                written += self._upsert_triple(
                    subject=user_id,
                    predicate=predicate,
                    object=target,
                    derived_from_path=derived_from_path,
                )
                self._upsert_provenance_edge(target_id, derived_from_path)
        return written

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
        """Pre-seed a triple and its DERIVED_FROM provenance edge.

        Used by tests that arrange graph state before calling regenerator/
        mark_orphaned methods. Writes both the typed triple record and a
        FakeDerivedFromEdge so mark_orphaned_by_provenance_path (which queries
        _provenance_edges) can find the triple.

        The provenance edge status mirrors the triple status: seeding an orphaned
        triple also seeds an orphaned provenance edge so get_orphaned_provenance_paths
        returns the path correctly.
        """
        self._triples.append(
            FakeTriple(
                subject=subject,
                predicate=predicate,
                object=object,
                derived_from_path=derived_from_path,
                status=status,
            )
        )
        if derived_from_path:
            entity_id = f"{subject}-{object}"
            key = (entity_id, derived_from_path)
            self._provenance_edges[key] = FakeDerivedFromEdge(
                entity_id=entity_id,
                path=derived_from_path,
                status=status,
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _upsert_triple(
        self,
        subject: str,
        predicate: str,
        object: str,
        derived_from_path: str,
    ) -> int:
        """Insert triple if not already present; returns 1 if written, 0 if exists."""
        existing = self.get_triple(subject, predicate, object)
        if existing is not None:
            # Update provenance path on re-derive; keep status (may be orphaned)
            existing.derived_from_path = derived_from_path
            existing.status = "active"
            return 0
        self._triples.append(
            FakeTriple(
                subject=subject,
                predicate=predicate,
                object=object,
                derived_from_path=derived_from_path,
                status="active",
            )
        )
        return 1

    def _upsert_provenance_edge(self, entity_id: str, path: str) -> None:
        """MERGE a DERIVED_FROM provenance edge (entity_id, path) -> VaultNote.

        Idempotent: re-upsert resets status to 'active' (mirrors ON MATCH SET).
        Keyed by (entity_id, path) so one edge per (entity, source) pair.
        """
        key = (entity_id, path)
        existing = self._provenance_edges.get(key)
        if existing is not None:
            existing.status = "active"
        else:
            self._provenance_edges[key] = FakeDerivedFromEdge(
                entity_id=entity_id,
                path=path,
                status="active",
            )
