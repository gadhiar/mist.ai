"""ADR-010 Cluster 8 Phase 10: vault bootstrap from seed_data.

Verifies that `bootstrap_vault_from_seed` writes identity/mist.md and
users/<id>.md and that `emit_seed_vault_provenance` MERGE-creates the
VaultNote nodes and DERIVED_FROM edges from each seeded entity.
"""

from __future__ import annotations

import pytest

from backend.knowledge import admin
from tests.mocks.neo4j import FakeNeo4jConnection

# Reuse the minimal seed shape from test_admin_seed.
SEED = {
    "ontology_version": "1.0.0",
    "mist_identity": {
        "id": "mist-identity",
        "entity_type": "MistIdentity",
        "display_name": "MIST",
        "pronouns": "she/her",
        "age_analog": "26-27",
        "self_concept": "test concept",
        "origin": "test origin",
        "baseline_persona_seeded": True,
        "growth_enabled": True,
        "version": "0.1.0-mvp",
    },
    "traits": [
        {"id": "trait-a", "display_name": "Trait A", "axis": "Persona", "description": "a"},
        {"id": "trait-b", "display_name": "Trait B", "axis": "Persona", "description": "b"},
    ],
    "capabilities": [{"id": "cap-a", "display_name": "Cap A", "description": "a"}],
    "preferences": [
        {
            "id": "pref-a",
            "display_name": "Pref A",
            "enforcement": "absolute",
            "context": "a",
        }
    ],
    "user": {
        "id": "user",
        "entity_type": "User",
        "display_name": "Raj Gadhia",
    },
    "entities": [
        {"id": "slalom", "entity_type": "Organization", "display_name": "Slalom"},
        {"id": "python", "entity_type": "Technology", "display_name": "Python"},
    ],
    "identity_relationships": [],
    "anchor_relationships": [],
}


# ---------------------------------------------------------------------------
# FakeAsyncVaultWriter -- records calls + returns canned paths
# ---------------------------------------------------------------------------


class FakeAsyncVaultWriter:
    """Minimal async vault writer recording upsert_identity/upsert_user calls."""

    def __init__(self, root: str = "/tmp/vault") -> None:
        self.root = root
        self.identity_calls: list[dict] = []
        self.user_calls: list[dict] = []

    async def upsert_identity(self, traits, capabilities, preferences) -> str:
        self.identity_calls.append(
            {"traits": traits, "capabilities": capabilities, "preferences": preferences}
        )
        return f"{self.root}/identity/mist.md"

    async def upsert_user(self, user_id: str, body_markdown: str) -> str:
        self.user_calls.append({"user_id": user_id, "body_markdown": body_markdown})
        return f"{self.root}/users/{user_id}.md"


# ---------------------------------------------------------------------------
# TestBuildUserBodyMarkdown
# ---------------------------------------------------------------------------


class TestBuildUserBodyMarkdown:
    def test_includes_display_name_as_h1(self) -> None:
        body = admin._build_user_body_markdown(SEED["user"])
        assert body.startswith("# Raj Gadhia")

    def test_includes_profile_section(self) -> None:
        body = admin._build_user_body_markdown(SEED["user"])
        assert "## Profile" in body

    def test_falls_back_to_id_when_display_name_missing(self) -> None:
        body = admin._build_user_body_markdown({"id": "user"})
        assert body.startswith("# user")

    def test_skips_structural_keys(self) -> None:
        body = admin._build_user_body_markdown(SEED["user"])
        # id / entity_type / display_name are structural, not body content
        assert "**id**" not in body
        assert "**entity_type**" not in body
        assert "**display_name**" not in body

    def test_renders_extra_scalar_properties(self) -> None:
        seed_with_extras = {
            "id": "u",
            "display_name": "X",
            "title": "Engineer",
            "city": "Chicago",
        }
        body = admin._build_user_body_markdown(seed_with_extras)
        assert "**title**: Engineer" in body
        assert "**city**: Chicago" in body

    def test_deterministic_ordering(self) -> None:
        # Same input -> same output across calls (alphabetical key order).
        body_1 = admin._build_user_body_markdown(SEED["user"])
        body_2 = admin._build_user_body_markdown(SEED["user"])
        assert body_1 == body_2


# ---------------------------------------------------------------------------
# TestBootstrapVaultFromSeed
# ---------------------------------------------------------------------------


class TestBootstrapVaultFromSeed:
    @pytest.mark.asyncio
    async def test_returns_both_paths(self) -> None:
        writer = FakeAsyncVaultWriter()

        paths = await admin.bootstrap_vault_from_seed(writer, SEED)

        assert "identity_path" in paths
        assert "user_path" in paths
        assert paths["identity_path"].endswith("/identity/mist.md")
        assert paths["user_path"].endswith("/users/user.md")

    @pytest.mark.asyncio
    async def test_forwards_traits_capabilities_preferences(self) -> None:
        writer = FakeAsyncVaultWriter()

        await admin.bootstrap_vault_from_seed(writer, SEED)

        assert len(writer.identity_calls) == 1
        call = writer.identity_calls[0]
        assert call["traits"] == SEED["traits"]
        assert call["capabilities"] == SEED["capabilities"]
        assert call["preferences"] == SEED["preferences"]

    @pytest.mark.asyncio
    async def test_user_call_carries_user_id_and_body(self) -> None:
        writer = FakeAsyncVaultWriter()

        await admin.bootstrap_vault_from_seed(writer, SEED)

        assert len(writer.user_calls) == 1
        call = writer.user_calls[0]
        assert call["user_id"] == "user"
        assert "Raj Gadhia" in call["body_markdown"]

    @pytest.mark.asyncio
    async def test_handles_empty_trait_lists(self) -> None:
        writer = FakeAsyncVaultWriter()
        seed = {**SEED, "traits": [], "capabilities": [], "preferences": []}

        paths = await admin.bootstrap_vault_from_seed(writer, seed)

        # upsert_identity still called (empty rendering is valid)
        assert len(writer.identity_calls) == 1
        assert writer.identity_calls[0]["traits"] == []
        assert paths["identity_path"]


# ---------------------------------------------------------------------------
# TestEmitSeedVaultProvenance
# ---------------------------------------------------------------------------


class TestEmitSeedVaultProvenance:
    def test_creates_two_vault_note_nodes(self) -> None:
        # Arrange
        conn = FakeNeo4jConnection()

        # Act
        admin.emit_seed_vault_provenance(
            conn,
            SEED,
            identity_path="/v/identity/mist.md",
            user_path="/v/users/user.md",
            now_iso="2026-04-22T00:00:00Z",
        )

        # Assert -- exactly two VaultNote node MERGE writes
        node_merges = [q for q, _ in conn.writes if "MERGE (vn:__Provenance__:VaultNote" in q]
        assert len(node_merges) == 2

    def test_emits_identity_edges_for_identity_traits_caps_prefs(self) -> None:
        # Arrange
        conn = FakeNeo4jConnection()

        # Act
        edges = admin.emit_seed_vault_provenance(
            conn,
            SEED,
            identity_path="/v/identity/mist.md",
            user_path="/v/users/user.md",
        )

        # Assert -- 1 identity + 2 traits + 1 capability + 1 preference = 5
        # plus 1 user + 2 anchor entities = 3
        # total = 8
        assert edges == 8

    def test_identity_entities_target_identity_path(self) -> None:
        # Arrange
        conn = FakeNeo4jConnection()

        # Act
        admin.emit_seed_vault_provenance(
            conn,
            SEED,
            identity_path="/v/identity/mist.md",
            user_path="/v/users/user.md",
        )

        # Assert -- mist-identity MERGE has identity_path target
        derived_writes = [(q, p) for q, p in conn.writes if "MERGE (e)-[r:DERIVED_FROM]->(vn)" in q]
        # Find the call for mist-identity entity
        mist_calls = [p for _, p in derived_writes if p.get("entity_id") == "mist-identity"]
        assert len(mist_calls) == 1
        assert mist_calls[0]["path"] == "/v/identity/mist.md"

    def test_user_entities_target_user_path(self) -> None:
        # Arrange
        conn = FakeNeo4jConnection()

        # Act
        admin.emit_seed_vault_provenance(
            conn,
            SEED,
            identity_path="/v/identity/mist.md",
            user_path="/v/users/user.md",
        )

        # Assert
        derived_writes = [(q, p) for q, p in conn.writes if "MERGE (e)-[r:DERIVED_FROM]->(vn)" in q]
        slalom_calls = [p for _, p in derived_writes if p.get("entity_id") == "slalom"]
        assert len(slalom_calls) == 1
        assert slalom_calls[0]["path"] == "/v/users/user.md"

    def test_edges_carry_seed_event_id(self) -> None:
        # Arrange -- seed-derived edges use event_id="seed" (literal in the
        # Cypher) rather than a per-turn UUID. Phase 8 ontology_version /
        # extraction_version / model_hash stamps are NOT applied to seed
        # edges by design (seed entities are deterministic by construction
        # via apply_seed; rebuild = re-run mist_admin seed).
        conn = FakeNeo4jConnection()

        # Act
        admin.emit_seed_vault_provenance(
            conn,
            SEED,
            identity_path="/v/identity/mist.md",
            user_path="/v/users/user.md",
        )

        # Assert -- 'seed' literal appears on every DERIVED_FROM edge query;
        # Phase 8 property names do not appear at all.
        for q, _ in conn.writes:
            if "MERGE (e)-[r:DERIVED_FROM]->(vn)" in q:
                assert "r.event_id = 'seed'" in q
                # Phase 8 stamps absent on seed edges
                assert "ontology_version" not in q
                assert "extraction_version" not in q
                assert "model_hash" not in q

    def test_idempotent_across_two_runs(self) -> None:
        # Arrange + Act -- run twice; both rely on Cypher MERGE for idempotency.
        # The number of MERGE calls doubles, but the underlying graph state is
        # unchanged (Cypher MERGE semantics).
        conn = FakeNeo4jConnection()
        first = admin.emit_seed_vault_provenance(
            conn, SEED, identity_path="/v/i.md", user_path="/v/u.md"
        )
        second = admin.emit_seed_vault_provenance(
            conn, SEED, identity_path="/v/i.md", user_path="/v/u.md"
        )

        # Assert -- same edge count returned both times
        assert first == second
        # Caller-side determinism: second run does not change the contract.
