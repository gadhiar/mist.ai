"""Unit tests for bucket1_reader (inverse of writer.py identity/user rendering).

Forward direction:
  - identity: VaultWriter._upsert_identity_sync produces identity/mist.md
  - user: render_user_snapshot_body produces the body for users/<user>.md

These tests parse the file formats that the forward direction emits.
"""

from pathlib import Path

import pytest

from backend.knowledge.curation.bucket1_reader import (
    Bucket1ParseError,
    parse_identity_file,
    parse_user_file,
)

# ---------------------------------------------------------------------------
# parse_identity_file
# ---------------------------------------------------------------------------


def test_parse_identity_file_extracts_traits(tmp_path: Path) -> None:
    """Traits parsed from the ## Traits section (bold-name, optional axis)."""
    p = tmp_path / "mist.md"
    p.write_text(
        "---\n"
        "type: mist-identity\n"
        "authored_by: user\n"
        "version: '1.0'\n"
        "last_updated: '2026-05-10'\n"
        "tags: [identity]\n"
        "---\n"
        "\n"
        "# MIST Identity\n"
        "\n"
        "## Traits\n"
        "- **warm-and-playful** (emotional-register) -- default interaction mode\n"
        "- **precise** -- factual accuracy\n"
        "\n"
        "## Capabilities\n"
        "- **code-generation** -- writes and reviews code\n"
        "\n"
        "## Preferences\n"
        "- **no-emojis** (absolute) -- never use unicode decorative characters\n"
        "\n"
        "## Provenance\n"
        "- source: scripts/seed_data.yaml\n"
        "- rendered_at: 2026-05-10T00:00:00+00:00\n",
        encoding="utf-8",
    )
    parsed = parse_identity_file(p)
    assert parsed.traits == ["warm-and-playful", "precise"]
    assert parsed.capabilities == ["code-generation"]
    assert len(parsed.preferences) == 1
    assert parsed.preferences[0].slug == "no-emojis"
    assert parsed.preferences[0].enforcement == "absolute"


def test_parse_identity_file_multiple_preferences(tmp_path: Path) -> None:
    """Multiple preferences parsed correctly, order preserved."""
    p = tmp_path / "mist.md"
    p.write_text(
        "---\n"
        "type: mist-identity\n"
        "authored_by: user\n"
        "version: '1.0'\n"
        "last_updated: '2026-05-10'\n"
        "tags: []\n"
        "---\n"
        "\n"
        "# MIST Identity\n"
        "\n"
        "## Traits\n"
        "\n"
        "## Capabilities\n"
        "\n"
        "## Preferences\n"
        "- **no-emojis** (absolute) -- never use decorative characters\n"
        "- **plain-text-indicators** (soft) -- use [DONE] not checkmarks\n"
        "\n"
        "## Provenance\n"
        "- source: seed\n",
        encoding="utf-8",
    )
    parsed = parse_identity_file(p)
    assert len(parsed.preferences) == 2
    assert parsed.preferences[0].slug == "no-emojis"
    assert parsed.preferences[0].enforcement == "absolute"
    assert parsed.preferences[1].slug == "plain-text-indicators"
    assert parsed.preferences[1].enforcement == "soft"


def test_parse_identity_file_empty_sections(tmp_path: Path) -> None:
    """Empty Traits/Capabilities sections return empty lists (not errors)."""
    p = tmp_path / "mist.md"
    p.write_text(
        "---\n"
        "type: mist-identity\n"
        "authored_by: user\n"
        "version: '1.0'\n"
        "last_updated: '2026-05-10'\n"
        "tags: []\n"
        "---\n"
        "\n"
        "# MIST Identity\n"
        "\n"
        "## Traits\n"
        "\n"
        "## Capabilities\n"
        "\n"
        "## Preferences\n"
        "\n"
        "## Provenance\n"
        "- source: seed\n",
        encoding="utf-8",
    )
    parsed = parse_identity_file(p)
    assert parsed.traits == []
    assert parsed.capabilities == []
    assert parsed.preferences == []


def test_parse_identity_file_missing_frontmatter_raises(tmp_path: Path) -> None:
    """No frontmatter block raises Bucket1ParseError."""
    p = tmp_path / "mist.md"
    p.write_text("# No frontmatter here\n", encoding="utf-8")
    with pytest.raises(Bucket1ParseError, match="frontmatter"):
        parse_identity_file(p)


def test_parse_identity_file_wrong_type_raises(tmp_path: Path) -> None:
    """Frontmatter with wrong type raises Bucket1ParseError."""
    p = tmp_path / "mist.md"
    p.write_text(
        "---\n" "type: mist-user\n" "user_id: raj\n" "---\n" "\n" "# Wrong type\n",
        encoding="utf-8",
    )
    with pytest.raises(Bucket1ParseError, match="mist-identity"):
        parse_identity_file(p)


def test_parse_identity_file_trait_without_axis(tmp_path: Path) -> None:
    """Trait bullets without axis parenthetical are still parsed correctly."""
    p = tmp_path / "mist.md"
    p.write_text(
        "---\n"
        "type: mist-identity\n"
        "authored_by: user\n"
        "version: '1.0'\n"
        "last_updated: '2026-05-10'\n"
        "tags: []\n"
        "---\n"
        "\n"
        "# MIST Identity\n"
        "\n"
        "## Traits\n"
        "- **calm** -- maintains composure\n"
        "- **precise** -- factual accuracy\n"
        "\n"
        "## Capabilities\n"
        "\n"
        "## Preferences\n"
        "\n"
        "## Provenance\n"
        "- source: seed\n",
        encoding="utf-8",
    )
    parsed = parse_identity_file(p)
    assert parsed.traits == ["calm", "precise"]


# ---------------------------------------------------------------------------
# parse_user_file
# ---------------------------------------------------------------------------


def test_parse_user_file_extracts_tools_and_interests(tmp_path: Path) -> None:
    """Tools and Technologies + Interests sections parsed correctly."""
    p = tmp_path / "raj.md"
    p.write_text(
        "---\n"
        "type: mist-user\n"
        "user_id: raj\n"
        "authored_by: mist\n"
        "last_updated: '2026-05-10'\n"
        "related_sessions: []\n"
        "tags: []\n"
        "---\n"
        "\n"
        "# Raj\n"
        "\n"
        "## Tools and Technologies\n"
        "- **Python** (Language) -- general-purpose programming\n"
        "- **Neo4j** (Database) -- graph database\n"
        "\n"
        "## Interests\n"
        "- **knowledge-graphs** (Topic)\n"
        "\n"
        "## Provenance\n"
        "- rendered_at: 2026-05-10T00:00:00+00:00\n"
        "- source: graph snapshot (User entity + 1-hop outbound neighbors)\n",
        encoding="utf-8",
    )
    parsed = parse_user_file(p)
    assert parsed.user_id == "raj"
    assert "Python" in parsed.tools_and_technologies
    assert "Neo4j" in parsed.tools_and_technologies
    assert "knowledge-graphs" in parsed.interests


def test_parse_user_file_multiple_sections(tmp_path: Path) -> None:
    """All major sections parsed into correct fields."""
    p = tmp_path / "raj.md"
    p.write_text(
        "---\n"
        "type: mist-user\n"
        "user_id: raj\n"
        "authored_by: mist\n"
        "last_updated: '2026-05-10'\n"
        "related_sessions: []\n"
        "tags: []\n"
        "---\n"
        "\n"
        "# Raj\n"
        "\n"
        "## Tools and Technologies\n"
        "- **Python** (Language)\n"
        "\n"
        "## Expertise\n"
        "- **distributed-systems** (Domain)\n"
        "\n"
        "## Currently Learning\n"
        "- **Rust** (Language)\n"
        "\n"
        "## Projects\n"
        "- **MIST.AI** (Project)\n"
        "\n"
        "## Affiliations\n"
        "- **Acme Corp** (Organization)\n"
        "\n"
        "## Interests\n"
        "- **knowledge-graphs** (Topic)\n"
        "\n"
        "## Goals\n"
        "- **ship-phase-3** (Goal)\n"
        "\n"
        "## People\n"
        "- **Alice** (Person)\n"
        "\n"
        "## Provenance\n"
        "- rendered_at: 2026-05-10T00:00:00+00:00\n",
        encoding="utf-8",
    )
    parsed = parse_user_file(p)
    assert "Python" in parsed.tools_and_technologies
    assert "distributed-systems" in parsed.expertise
    assert "Rust" in parsed.currently_learning
    assert "MIST.AI" in parsed.projects
    assert "Acme Corp" in parsed.affiliations
    assert "knowledge-graphs" in parsed.interests
    assert "ship-phase-3" in parsed.goals
    assert "Alice" in parsed.people


def test_parse_user_file_missing_frontmatter_raises(tmp_path: Path) -> None:
    """No frontmatter block raises Bucket1ParseError."""
    p = tmp_path / "raj.md"
    p.write_text("# Raj\n\nSome content.\n", encoding="utf-8")
    with pytest.raises(Bucket1ParseError, match="frontmatter"):
        parse_user_file(p)


def test_parse_user_file_wrong_type_raises(tmp_path: Path) -> None:
    """Wrong type in frontmatter raises Bucket1ParseError."""
    p = tmp_path / "raj.md"
    p.write_text(
        "---\n"
        "type: mist-identity\n"
        "authored_by: user\n"
        "version: '1.0'\n"
        "last_updated: '2026-05-10'\n"
        "tags: []\n"
        "---\n"
        "\n"
        "# Raj\n",
        encoding="utf-8",
    )
    with pytest.raises(Bucket1ParseError, match="mist-user"):
        parse_user_file(p)


def test_parse_user_file_missing_user_id_raises(tmp_path: Path) -> None:
    """Missing user_id in frontmatter raises Bucket1ParseError."""
    p = tmp_path / "raj.md"
    p.write_text(
        "---\n"
        "type: mist-user\n"
        "authored_by: mist\n"
        "last_updated: '2026-05-10'\n"
        "tags: []\n"
        "---\n"
        "\n"
        "# Raj\n",
        encoding="utf-8",
    )
    with pytest.raises(Bucket1ParseError, match="user_id"):
        parse_user_file(p)


def test_parse_user_file_empty_sections_return_empty_lists(tmp_path: Path) -> None:
    """User file with no neighbor sections returns empty lists (not errors)."""
    p = tmp_path / "raj.md"
    p.write_text(
        "---\n"
        "type: mist-user\n"
        "user_id: raj\n"
        "authored_by: mist\n"
        "last_updated: '2026-05-10'\n"
        "related_sessions: []\n"
        "tags: []\n"
        "---\n"
        "\n"
        "# Raj\n"
        "\n"
        "## Provenance\n"
        "- rendered_at: 2026-05-10T00:00:00+00:00\n",
        encoding="utf-8",
    )
    parsed = parse_user_file(p)
    assert parsed.tools_and_technologies == []
    assert parsed.interests == []
    assert parsed.expertise == []


# ---------------------------------------------------------------------------
# Round-trip: render_user_snapshot_body -> write file -> parse_user_file
# ---------------------------------------------------------------------------


def test_parse_user_file_round_trip_with_render_user_snapshot_body(tmp_path: Path) -> None:
    """render_user_snapshot_body output is correctly parsed by parse_user_file.

    This is the idempotency test: the inverse must recover the display_names
    present in the snapshot from the file that the forward renderer emits.
    """
    from backend.vault.models import AuthoredBy, MistUserFrontmatter, render_frontmatter
    from backend.vault.user_snapshot import NeighborRef, UserSnapshot, render_user_snapshot_body

    snapshot = UserSnapshot(
        user_id="raj",
        display_name="Raj",
        profile_attrs={},
        edges_by_type={
            "USES": [
                NeighborRef(
                    entity_id="python",
                    entity_type="Language",
                    display_name="Python",
                ),
                NeighborRef(
                    entity_id="neo4j",
                    entity_type="Database",
                    display_name="Neo4j",
                ),
            ],
            "INTERESTED_IN": [
                NeighborRef(
                    entity_id="knowledge-graphs",
                    entity_type="Topic",
                    display_name="knowledge-graphs",
                ),
            ],
            "LEARNING": [
                NeighborRef(
                    entity_id="rust",
                    entity_type="Language",
                    display_name="Rust",
                ),
            ],
        },
        rendered_at="2026-05-10T00:00:00+00:00",
    )
    body = render_user_snapshot_body(snapshot)
    fm = MistUserFrontmatter(
        user_id="raj",
        authored_by=AuthoredBy.MIST,
        last_updated="2026-05-10",
    )
    file_content = render_frontmatter(fm, body)

    p = tmp_path / "raj.md"
    p.write_text(file_content, encoding="utf-8")

    parsed = parse_user_file(p)

    assert parsed.user_id == "raj"
    assert "Python" in parsed.tools_and_technologies
    assert "Neo4j" in parsed.tools_and_technologies
    assert "knowledge-graphs" in parsed.interests
    assert "Rust" in parsed.currently_learning
