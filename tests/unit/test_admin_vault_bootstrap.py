"""ADR-010 Cluster 8 Phase 10: vault bootstrap from seed_data.

Verifies that `bootstrap_vault_from_seed` writes identity/mist.md and
users/<id>.md. R1.4 Task 6 retired the DERIVED_FROM->VaultNote seed
provenance edge (`emit_seed_vault_provenance`, formerly tested here) --
seed facts now carry a `seed_version` property instead. See
`tests/unit/knowledge/seed/test_no_vaultnote_provenance.py` for the
mutation-proof guard against reintroduction.
"""

from __future__ import annotations

import pytest

from backend.knowledge import admin

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

    async def upsert_identity(self, traits, capabilities, preferences, rendered_at=None) -> str:
        self.identity_calls.append(
            {
                "traits": traits,
                "capabilities": capabilities,
                "preferences": preferences,
                "rendered_at": rendered_at,
            }
        )
        return f"{self.root}/identity/mist.md"

    async def upsert_user(self, user_id: str, body_markdown: str, rendered_at=None) -> str:
        self.user_calls.append(
            {"user_id": user_id, "body_markdown": body_markdown, "rendered_at": rendered_at}
        )
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
