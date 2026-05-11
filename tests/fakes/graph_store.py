"""FakeGraphStore — in-memory test double for GraphStoreProtocol.

Satisfies the GraphStoreProtocol surface required by GraphRegenerator:
  - mark_orphaned_by_provenance_path
  - current_ontology_version
  - upsert_identity
  - upsert_user

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
    """In-memory test double for GraphStoreProtocol.

    Tracks triples in a list and records method call history.
    Idempotent upsert: calling upsert_identity / upsert_user with
    the same display_name twice writes only one triple (dedup by key).

    Supports `get_orphaned_provenance_paths` for retry_orphaned tests:
    returns the distinct set of derived_from_path values among triples
    whose status == 'orphaned'.
    """

    _ONTOLOGY_VERSION = "1.1.0"

    def __init__(self) -> None:
        self._triples: list[FakeTriple] = []
        self.mark_orphaned_calls: list[str] = []
        self.upsert_identity_calls: list[dict] = []
        self.upsert_user_calls: list[dict] = []

    # ------------------------------------------------------------------
    # GraphStoreProtocol methods
    # ------------------------------------------------------------------

    async def mark_orphaned_by_provenance_path(self, path: str) -> int:
        """Mark all triples with derived_from_path == path as 'orphaned'."""
        self.mark_orphaned_calls.append(path)
        count = 0
        for triple in self._triples:
            if triple.derived_from_path == path and triple.status != "orphaned":
                triple.status = "orphaned"
                count += 1
        return count

    def current_ontology_version(self) -> str:
        return self._ONTOLOGY_VERSION

    async def get_orphaned_provenance_paths(self) -> list[str]:
        """Return distinct derived_from_path values for orphaned triples."""
        seen: set[str] = set()
        result: list[str] = []
        for triple in self._triples:
            if triple.status == "orphaned" and triple.derived_from_path not in seen:
                seen.add(triple.derived_from_path)
                result.append(triple.derived_from_path)
        return result

    async def upsert_identity(self, parsed_identity, derived_from_path: str) -> int:
        """Write ParsedIdentity attributes as graph triples (idempotent)."""
        self.upsert_identity_calls.append(
            {"parsed_identity": parsed_identity, "derived_from_path": derived_from_path}
        )
        written = 0
        for trait_slug in parsed_identity.traits:
            written += self._upsert_triple(
                subject="mist-identity",
                predicate="HAS_TRAIT",
                object=trait_slug,
                derived_from_path=derived_from_path,
            )
        for cap_slug in parsed_identity.capabilities:
            written += self._upsert_triple(
                subject="mist-identity",
                predicate="HAS_CAPABILITY",
                object=cap_slug,
                derived_from_path=derived_from_path,
            )
        for pref in parsed_identity.preferences:
            written += self._upsert_triple(
                subject="mist-identity",
                predicate="HAS_PREFERENCE",
                object=pref.slug,
                derived_from_path=derived_from_path,
            )
        return written

    async def upsert_user(self, parsed_user, derived_from_path: str) -> int:
        """Write ParsedUser edge targets as graph triples (idempotent)."""
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
                written += self._upsert_triple(
                    subject=user_id,
                    predicate=predicate,
                    object=target,
                    derived_from_path=derived_from_path,
                )
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
        """Pre-seed a triple (for testing orphan-mark behavior)."""
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
