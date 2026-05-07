"""Tests for the user.md graph snapshot renderer + query.

Per ADR-011 bucket 1 (Mechanical state mirror) and the 2026-05-06 canonical
vault pattern: users/<user_id>.md is rendered from a 1-hop graph snapshot of
the User entity + its outbound neighbors, grouped by edge type into named
sections. Re-rendered after extraction touches user-scope; skipped otherwise.
"""

from __future__ import annotations

from backend.vault.user_snapshot import (
    NeighborRef,
    UserSnapshot,
    extraction_touched_user_scope,
    render_user_snapshot_body,
)


def _make_neighbor(
    entity_id: str = "python",
    entity_type: str = "Technology",
    display_name: str = "Python",
    description: str | None = None,
    confidence: float = 1.0,
) -> NeighborRef:
    return NeighborRef(
        entity_id=entity_id,
        entity_type=entity_type,
        display_name=display_name,
        description=description,
        confidence=confidence,
    )


class TestRendererStructure:
    def test_renders_h1_with_display_name(self):
        snap = UserSnapshot(
            user_id="user",
            display_name="Raj Gadhia",
            profile_attrs={},
            edges_by_type={},
            rendered_at="2026-05-06T22:00:00+00:00",
        )
        body = render_user_snapshot_body(snap)
        assert body.startswith("# Raj Gadhia\n")

    def test_falls_back_to_user_id_when_no_display_name(self):
        snap = UserSnapshot(
            user_id="raj",
            display_name=None,
            profile_attrs={},
            edges_by_type={},
            rendered_at="2026-05-06T22:00:00+00:00",
        )
        body = render_user_snapshot_body(snap)
        assert body.startswith("# raj\n")

    def test_provenance_section_always_present(self):
        snap = UserSnapshot(
            user_id="user",
            display_name="Raj",
            profile_attrs={},
            edges_by_type={},
            rendered_at="2026-05-06T22:00:00+00:00",
        )
        body = render_user_snapshot_body(snap)
        assert "## Provenance" in body
        assert "rendered_at: 2026-05-06T22:00:00+00:00" in body


class TestProfileSection:
    def test_profile_attrs_listed_as_bullets(self):
        snap = UserSnapshot(
            user_id="user",
            display_name="Raj Gadhia",
            profile_attrs={"location": "Detroit", "role": "Software Engineer"},
            edges_by_type={},
            rendered_at="2026-05-06T22:00:00+00:00",
        )
        body = render_user_snapshot_body(snap)
        assert "## Profile" in body
        assert "- location: Detroit" in body
        assert "- role: Software Engineer" in body

    def test_profile_section_omitted_when_empty(self):
        snap = UserSnapshot(
            user_id="user",
            display_name="Raj",
            profile_attrs={},
            edges_by_type={},
            rendered_at="2026-05-06T22:00:00+00:00",
        )
        body = render_user_snapshot_body(snap)
        assert "## Profile" not in body


class TestEdgeSections:
    """Per ADR-011 + canonical pattern memo: each section maps to specific edge types."""

    def test_uses_renders_under_tools_and_technologies(self):
        snap = UserSnapshot(
            user_id="user",
            display_name="Raj",
            profile_attrs={},
            edges_by_type={
                "USES": [_make_neighbor(entity_id="python", display_name="Python")],
            },
            rendered_at="2026-05-06T22:00:00+00:00",
        )
        body = render_user_snapshot_body(snap)
        assert "## Tools and Technologies" in body
        assert "**Python**" in body

    def test_works_with_and_depends_on_share_tools_section(self):
        snap = UserSnapshot(
            user_id="user",
            display_name="Raj",
            profile_attrs={},
            edges_by_type={
                "USES": [_make_neighbor("neo4j", display_name="Neo4j")],
                "WORKS_WITH": [_make_neighbor("docker", display_name="Docker")],
                "DEPENDS_ON": [_make_neighbor("cuda", display_name="CUDA")],
            },
            rendered_at="2026-05-06T22:00:00+00:00",
        )
        body = render_user_snapshot_body(snap)
        # All three edge types collapse into one Tools section
        tools_section_count = body.count("## Tools and Technologies")
        assert tools_section_count == 1
        assert "**Neo4j**" in body
        assert "**Docker**" in body
        assert "**CUDA**" in body

    def test_expert_in_renders_under_expertise(self):
        snap = UserSnapshot(
            user_id="user",
            display_name="Raj",
            profile_attrs={},
            edges_by_type={
                "EXPERT_IN": [
                    _make_neighbor("software-engineering", "Skill", "Software Engineering"),
                ],
            },
            rendered_at="2026-05-06T22:00:00+00:00",
        )
        body = render_user_snapshot_body(snap)
        assert "## Expertise" in body
        assert "**Software Engineering**" in body

    def test_learning_renders_under_currently_learning(self):
        snap = UserSnapshot(
            user_id="user",
            display_name="Raj",
            profile_attrs={},
            edges_by_type={
                "LEARNING": [_make_neighbor("rust", "Technology", "Rust")],
            },
            rendered_at="2026-05-06T22:00:00+00:00",
        )
        body = render_user_snapshot_body(snap)
        assert "## Currently Learning" in body
        assert "**Rust**" in body

    def test_works_on_renders_under_projects(self):
        snap = UserSnapshot(
            user_id="user",
            display_name="Raj",
            profile_attrs={},
            edges_by_type={
                "WORKS_ON": [_make_neighbor("mist-ai", "Project", "MIST.AI")],
            },
            rendered_at="2026-05-06T22:00:00+00:00",
        )
        body = render_user_snapshot_body(snap)
        assert "## Projects" in body
        assert "**MIST.AI**" in body

    def test_works_at_renders_under_affiliations(self):
        snap = UserSnapshot(
            user_id="user",
            display_name="Raj",
            profile_attrs={},
            edges_by_type={
                "WORKS_AT": [_make_neighbor("slalom", "Organization", "Slalom")],
            },
            rendered_at="2026-05-06T22:00:00+00:00",
        )
        body = render_user_snapshot_body(snap)
        assert "## Affiliations" in body
        assert "**Slalom**" in body

    def test_interested_in_renders_under_interests(self):
        snap = UserSnapshot(
            user_id="user",
            display_name="Raj",
            profile_attrs={},
            edges_by_type={
                "INTERESTED_IN": [
                    _make_neighbor("cognitive-architecture", "Concept", "Cognitive Architecture"),
                ],
            },
            rendered_at="2026-05-06T22:00:00+00:00",
        )
        body = render_user_snapshot_body(snap)
        assert "## Interests" in body
        assert "**Cognitive Architecture**" in body

    def test_has_goal_renders_under_goals(self):
        snap = UserSnapshot(
            user_id="user",
            display_name="Raj",
            profile_attrs={},
            edges_by_type={
                "HAS_GOAL": [
                    _make_neighbor(
                        "persistent-memory", "Goal", "Persistent Memory Across Sessions"
                    ),
                ],
            },
            rendered_at="2026-05-06T22:00:00+00:00",
        )
        body = render_user_snapshot_body(snap)
        assert "## Goals" in body
        assert "**Persistent Memory Across Sessions**" in body

    def test_prefers_and_dislikes_share_preferences_section(self):
        snap = UserSnapshot(
            user_id="user",
            display_name="Raj",
            profile_attrs={},
            edges_by_type={
                "PREFERS": [_make_neighbor("local-first", "Concept", "Local-First")],
                "DISLIKES": [_make_neighbor("cloud-only", "Concept", "Cloud-Only")],
            },
            rendered_at="2026-05-06T22:00:00+00:00",
        )
        body = render_user_snapshot_body(snap)
        assert body.count("## Preferences") == 1
        assert "**Local-First**" in body
        assert "**Cloud-Only**" in body

    def test_knows_person_renders_under_people(self):
        snap = UserSnapshot(
            user_id="user",
            display_name="Raj",
            profile_attrs={},
            edges_by_type={
                "KNOWS_PERSON": [_make_neighbor("alex", "Person", "Alex Park")],
            },
            rendered_at="2026-05-06T22:00:00+00:00",
        )
        body = render_user_snapshot_body(snap)
        assert "## People" in body
        assert "**Alex Park**" in body


class TestEmptySectionOmission:
    def test_no_edge_types_means_no_section_headers(self):
        snap = UserSnapshot(
            user_id="user",
            display_name="Raj",
            profile_attrs={},
            edges_by_type={},
            rendered_at="2026-05-06T22:00:00+00:00",
        )
        body = render_user_snapshot_body(snap)
        # Only H1 + Provenance should appear; no edge section headers
        assert "## Tools and Technologies" not in body
        assert "## Expertise" not in body
        assert "## Currently Learning" not in body
        assert "## Projects" not in body
        assert "## Affiliations" not in body
        assert "## Interests" not in body
        assert "## Goals" not in body
        assert "## Preferences" not in body
        assert "## People" not in body

    def test_unknown_edge_type_is_ignored(self):
        """Unknown edge types do not produce a section.

        Edge types not in the canonical mapping (e.g., HAS_TRAIT, RELATED_TO,
        cross-layer plumbing edges that should never reach the renderer) are
        silently dropped. The query layer is responsible for filtering;
        the renderer is forgiving.
        """
        snap = UserSnapshot(
            user_id="user",
            display_name="Raj",
            profile_attrs={},
            edges_by_type={
                "USES": [_make_neighbor("python", display_name="Python")],
                "RELATED_TO": [_make_neighbor("cooking", "Concept", "Cooking")],
            },
            rendered_at="2026-05-06T22:00:00+00:00",
        )
        body = render_user_snapshot_body(snap)
        assert "**Python**" in body
        assert "**Cooking**" not in body


class TestExtractionTouchedUserScope:
    """C-pattern trigger detector. Returns True iff extraction touched User."""

    def test_returns_false_for_empty_extraction(self):
        assert extraction_touched_user_scope([], []) is False

    def test_returns_true_when_user_entity_extracted(self):
        entities = [{"entity_id": "user", "entity_type": "User", "display_name": "Raj"}]
        assert extraction_touched_user_scope(entities, []) is True

    def test_returns_true_when_user_entity_under_alternative_id_key(self):
        """Some pipelines emit `id` instead of `entity_id`; both should trigger."""
        entities = [{"id": "user", "entity_type": "User"}]
        assert extraction_touched_user_scope(entities, []) is True

    def test_returns_true_when_user_is_relationship_source(self):
        rels = [{"source": "user", "target": "python", "type": "USES"}]
        assert extraction_touched_user_scope([], rels) is True

    def test_returns_true_when_user_is_relationship_target(self):
        rels = [{"source": "alex", "target": "user", "type": "KNOWS_PERSON"}]
        assert extraction_touched_user_scope([], rels) is True

    def test_returns_false_when_no_user_involvement(self):
        entities = [{"entity_id": "neo4j", "entity_type": "Technology"}]
        rels = [{"source": "mist-identity", "target": "neo4j", "type": "USES"}]
        assert extraction_touched_user_scope(entities, rels) is False

    def test_respects_custom_user_id(self):
        """Multi-user MIST: user_id may not be 'user'."""
        entities = [{"entity_id": "raj", "entity_type": "User"}]
        assert extraction_touched_user_scope(entities, [], user_id="raj") is True
        assert extraction_touched_user_scope(entities, [], user_id="user") is False


class TestDeterminism:
    """Bytewise determinism is required for sidecar reindex efficiency.

    Same input -> same output, every time. No timestamps in the body other
    than rendered_at (which is part of the input).
    """

    def test_same_input_produces_byte_identical_output(self):
        snap = UserSnapshot(
            user_id="user",
            display_name="Raj",
            profile_attrs={"location": "Detroit"},
            edges_by_type={
                "USES": [_make_neighbor("python", display_name="Python")],
            },
            rendered_at="2026-05-06T22:00:00+00:00",
        )
        body_a = render_user_snapshot_body(snap)
        body_b = render_user_snapshot_body(snap)
        assert body_a == body_b

    def test_neighbors_sorted_alphabetically_within_section(self):
        """Stable ordering within a section: alphabetical by display_name."""
        snap = UserSnapshot(
            user_id="user",
            display_name="Raj",
            profile_attrs={},
            edges_by_type={
                "USES": [
                    _make_neighbor("zsh", "Technology", "Zsh"),
                    _make_neighbor("apple", "Technology", "Apple"),
                    _make_neighbor("middle", "Technology", "Middle"),
                ],
            },
            rendered_at="2026-05-06T22:00:00+00:00",
        )
        body = render_user_snapshot_body(snap)
        idx_apple = body.find("**Apple**")
        idx_middle = body.find("**Middle**")
        idx_zsh = body.find("**Zsh**")
        assert idx_apple < idx_middle < idx_zsh, "expected alphabetical order Apple, Middle, Zsh"
