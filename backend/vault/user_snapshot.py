"""User-vault graph snapshot renderer + query.

Per ADR-011 bucket 1 (Mechanical state mirror): `users/<user_id>.md` is a
1-hop graph snapshot of the User entity + its outbound neighbors, grouped
by edge type into named sections. Re-rendered after extraction touches
user-scope (User-source/target edges or User attribute changes); skipped
otherwise.

The renderer is a pure function over `UserSnapshot`. The query function
reads the graph and constructs the snapshot. Two-layer design lets us
unit-test the renderer in isolation from Neo4j.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class NeighborRef:
    """One outbound 1-hop neighbor of the User entity."""

    entity_id: str
    entity_type: str
    display_name: str
    description: str | None = None
    confidence: float = 1.0


@dataclass(frozen=True, slots=True)
class UserSnapshot:
    """Snapshot of the User entity + 1-hop neighbors at render time.

    Produced by `query_user_snapshot`; consumed by `render_user_snapshot_body`.
    """

    user_id: str
    display_name: str | None
    profile_attrs: dict[str, str]
    edges_by_type: dict[str, list[NeighborRef]]
    rendered_at: str  # ISO 8601 string; caller-supplied for determinism in tests


# Edge-type -> section-label mapping. Edges not in this map are silently
# omitted from the rendered body (the query layer is responsible for
# filtering cross-layer plumbing edges; the renderer is forgiving).
#
# Section order in the output follows the order of first appearance in
# this mapping (Python dicts preserve insertion order since 3.7).
_EDGE_TO_SECTION: dict[str, str] = {
    "USES": "Tools and Technologies",
    "WORKS_WITH": "Tools and Technologies",
    "DEPENDS_ON": "Tools and Technologies",
    "EXPERT_IN": "Expertise",
    "KNOWS": "Expertise",
    "LEARNING": "Currently Learning",
    "STRUGGLES_WITH": "Currently Learning",
    "WORKS_ON": "Projects",
    "WORKS_AT": "Affiliations",
    "MEMBER_OF": "Affiliations",
    "INTERESTED_IN": "Interests",
    "HAS_GOAL": "Goals",
    "PREFERS": "Preferences",
    "DISLIKES": "Preferences",
    "KNOWS_PERSON": "People",
}


def render_user_snapshot_body(snapshot: UserSnapshot) -> str:
    """Render a UserSnapshot to the markdown body for `users/<user_id>.md`.

    Pure function -- same input always produces byte-identical output.
    Empty sections are omitted. Neighbors within a section are sorted
    alphabetically by `display_name` for stable ordering.

    The body does NOT include frontmatter; the VaultWriter's `upsert_user`
    handles the frontmatter on top of this body.
    """
    parts: list[str] = []

    title = snapshot.display_name or snapshot.user_id
    parts.append(f"# {title}\n")

    if snapshot.profile_attrs:
        parts.append("\n## Profile\n")
        for key in sorted(snapshot.profile_attrs):
            parts.append(f"- {key}: {snapshot.profile_attrs[key]}\n")

    # Group neighbors by section label, preserving order from _EDGE_TO_SECTION.
    section_neighbors: dict[str, list[NeighborRef]] = {}
    for edge_type, neighbors in snapshot.edges_by_type.items():
        section = _EDGE_TO_SECTION.get(edge_type)
        if section is None:
            continue
        section_neighbors.setdefault(section, []).extend(neighbors)

    # Determine section order from first occurrence in the canonical mapping.
    seen_sections: list[str] = []
    for edge_type in _EDGE_TO_SECTION:
        section = _EDGE_TO_SECTION[edge_type]
        if section in section_neighbors and section not in seen_sections:
            seen_sections.append(section)

    for section in seen_sections:
        parts.append(f"\n## {section}\n")
        # Stable ordering: alphabetical by display_name.
        sorted_neighbors = sorted(section_neighbors[section], key=lambda n: n.display_name.lower())
        for n in sorted_neighbors:
            line = f"- **{n.display_name}** ({n.entity_type})"
            if n.description:
                line += f" -- {n.description}"
            parts.append(line + "\n")

    parts.append("\n## Provenance\n")
    parts.append(f"- rendered_at: {snapshot.rendered_at}\n")
    parts.append("- source: graph snapshot (User entity + 1-hop outbound neighbors)\n")

    return "".join(parts)


# ---------------------------------------------------------------------------
# Graph query
# ---------------------------------------------------------------------------


# Cross-layer plumbing edges that span Entity <-> Provenance and should
# never appear in a user.md snapshot regardless of source/target type.
_PROVENANCE_REL_TYPES: frozenset[str] = frozenset(
    {
        "DERIVED_FROM",
        "EXTRACTED_FROM",
        "ABOUT",
        "LEARNED_FROM",
        "SOURCED_FROM",
    }
)


async def query_user_snapshot(graph_executor, user_id: str, rendered_at: str) -> UserSnapshot:
    """Query the graph for the User entity + 1-hop outbound neighbors.

    Filters out cross-layer plumbing edges (DERIVED_FROM, EXTRACTED_FROM,
    ABOUT, LEARNED_FROM, SOURCED_FROM) -- those exist for graph rebuildability,
    not for user.md content.

    Args:
        graph_executor: GraphExecutor for async Cypher reads.
        user_id: The User entity id (typically "user" for single-user MIST).
        rendered_at: ISO 8601 timestamp string for the snapshot's
            Provenance section. Caller-supplied so tests are deterministic.

    Returns:
        UserSnapshot ready for `render_user_snapshot_body`.
    """
    rows = await graph_executor.execute_query(
        "MATCH (u:__Entity__ {id: $user_id}) "
        "RETURN u.display_name AS display_name, "
        "       properties(u) AS attrs",
        {"user_id": user_id},
    )
    if not rows:
        return UserSnapshot(
            user_id=user_id,
            display_name=None,
            profile_attrs={},
            edges_by_type={},
            rendered_at=rendered_at,
        )
    row = rows[0]
    display_name = row.get("display_name")
    raw_attrs = row.get("attrs") or {}
    profile_attrs: dict[str, str] = {}
    _PROFILE_FIELDS = {"location", "role", "title", "bio"}
    for k, v in raw_attrs.items():
        if k in _PROFILE_FIELDS and v is not None:
            profile_attrs[k] = str(v)

    neighbor_rows = await graph_executor.execute_query(
        "MATCH (u:__Entity__ {id: $user_id})-[r]->(n:__Entity__) "
        "RETURN type(r) AS rel_type, "
        "       n.id AS entity_id, "
        "       n.entity_type AS entity_type, "
        "       coalesce(n.display_name, n.id) AS display_name, "
        "       n.description AS description, "
        "       coalesce(r.confidence, 1.0) AS confidence",
        {"user_id": user_id},
    )
    edges_by_type: dict[str, list[NeighborRef]] = {}
    for nr in neighbor_rows:
        rel_type = nr.get("rel_type") or ""
        if rel_type in _PROVENANCE_REL_TYPES:
            continue
        ref = NeighborRef(
            entity_id=nr.get("entity_id") or "",
            entity_type=nr.get("entity_type") or "",
            display_name=nr.get("display_name") or nr.get("entity_id") or "",
            description=nr.get("description"),
            confidence=float(nr.get("confidence") or 1.0),
        )
        edges_by_type.setdefault(rel_type, []).append(ref)

    return UserSnapshot(
        user_id=user_id,
        display_name=display_name,
        profile_attrs=profile_attrs,
        edges_by_type=edges_by_type,
        rendered_at=rendered_at,
    )


def extraction_touched_user_scope(
    entities: list[dict],
    relationships: list[dict],
    user_id: str = "user",
) -> bool:
    """Detect whether an extraction batch touched user-scope.

    User-scope = the User entity itself, OR any relationship with User as
    source or target. C-pattern trigger for re-rendering `users/<user_id>.md`
    per ADR-011.

    Args:
        entities: List of extracted entity dicts (post-validation).
        relationships: List of extracted relationship dicts (post-validation).
        user_id: The User entity id (default "user").

    Returns:
        True if any extracted entity has id=user_id, or any relationship has
        source/target=user_id. False otherwise.
    """
    for ent in entities:
        if ent.get("entity_id") == user_id or ent.get("id") == user_id:
            return True
    for rel in relationships:
        if rel.get("source") == user_id or rel.get("target") == user_id:
            return True
    return False
