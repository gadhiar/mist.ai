"""R1.4 Task 10: vault bootstrap from the versioned seed source.

Verifies that `bootstrap_vault_from_seed` writes identity/mist.md and
users/<id>.md from `documents: list[SeedDocument]` -- the retired
`scripts/seed_data.yaml` dict path (and its `_build_user_body_markdown`
renderer, deleted this task) is gone. Each document's body is written
VERBATIM; there is no structured-field rendering left on this path (that
remains `upsert_identity`'s job, which has no production caller after this
task -- see `backend/vault/writer.py`).

R1.4 Task 6 retired the DERIVED_FROM->VaultNote seed provenance edge
(`emit_seed_vault_provenance`, formerly tested here) -- seed facts now carry
a `seed_version` property instead. See
`tests/unit/knowledge/seed/test_no_vaultnote_provenance.py` for the
mutation-proof guard against reintroduction.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.errors import SeedSourceError
from backend.knowledge import admin
from backend.knowledge.seed.models import SeedDocument
from backend.knowledge.storage.partitions import ENTITY_LABEL, SELF_MODEL_LABEL

_IDENTITY_DOC = SeedDocument(
    seed_version="profile-v1",
    facts=[],
    body="# MIST Identity\n\n## Traits\n- **Warm** -- Default register is warm.\n",
    source_path=Path("mist-memory/seed/mist.md"),
    partition=SELF_MODEL_LABEL,
)
_USER_DOC = SeedDocument(
    seed_version="profile-v1",
    facts=[],
    body="# user\n\n## Professional and Projects\nRaj Gadhia is a Software Engineer.\n",
    source_path=Path("mist-memory/seed/user.md"),
    partition=ENTITY_LABEL,
)
_DOCUMENTS = [_IDENTITY_DOC, _USER_DOC]


# ---------------------------------------------------------------------------
# FakeAsyncVaultWriter -- records calls + returns canned paths
# ---------------------------------------------------------------------------


class FakeAsyncVaultWriter:
    """Minimal async vault writer recording upsert_identity_body/upsert_user calls."""

    def __init__(self, root: str = "/tmp/vault") -> None:
        self.root = root
        self.identity_calls: list[dict] = []
        self.user_calls: list[dict] = []

    async def upsert_identity_body(
        self, body_markdown: str, source_path: str, rendered_at=None
    ) -> str:
        self.identity_calls.append(
            {
                "body_markdown": body_markdown,
                "source_path": source_path,
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
# TestBootstrapVaultFromSeed
# ---------------------------------------------------------------------------


class TestBootstrapVaultFromSeed:
    @pytest.mark.asyncio
    async def test_returns_both_paths(self) -> None:
        writer = FakeAsyncVaultWriter()

        paths = await admin.bootstrap_vault_from_seed(writer, _DOCUMENTS)

        assert "identity_path" in paths
        assert "user_path" in paths
        assert paths["identity_path"].endswith("/identity/mist.md")
        assert paths["user_path"].endswith("/users/user.md")

    @pytest.mark.asyncio
    async def test_identity_call_carries_body_and_source_verbatim(self) -> None:
        writer = FakeAsyncVaultWriter()

        await admin.bootstrap_vault_from_seed(writer, _DOCUMENTS)

        assert len(writer.identity_calls) == 1
        call = writer.identity_calls[0]
        assert call["body_markdown"] == _IDENTITY_DOC.body
        assert call["source_path"] == str(_IDENTITY_DOC.source_path)

    @pytest.mark.asyncio
    async def test_user_call_carries_user_id_from_source_stem_and_body_verbatim(self) -> None:
        writer = FakeAsyncVaultWriter()

        await admin.bootstrap_vault_from_seed(writer, _DOCUMENTS)

        assert len(writer.user_calls) == 1
        call = writer.user_calls[0]
        assert call["user_id"] == "user"  # source_path.stem, not a frontmatter field
        assert call["body_markdown"] == _USER_DOC.body

    @pytest.mark.asyncio
    async def test_rendered_at_threaded_to_both_calls(self) -> None:
        writer = FakeAsyncVaultWriter()

        await admin.bootstrap_vault_from_seed(
            writer, _DOCUMENTS, rendered_at="2026-05-07T00:00:00+00:00"
        )

        assert writer.identity_calls[0]["rendered_at"] == "2026-05-07T00:00:00+00:00"
        assert writer.user_calls[0]["rendered_at"] == "2026-05-07T00:00:00+00:00"

    @pytest.mark.asyncio
    async def test_raises_when_no_self_model_document(self) -> None:
        writer = FakeAsyncVaultWriter()

        with pytest.raises(SeedSourceError, match=SELF_MODEL_LABEL):
            await admin.bootstrap_vault_from_seed(writer, [_USER_DOC])

    @pytest.mark.asyncio
    async def test_raises_when_no_entity_document(self) -> None:
        writer = FakeAsyncVaultWriter()

        with pytest.raises(SeedSourceError, match=ENTITY_LABEL):
            await admin.bootstrap_vault_from_seed(writer, [_IDENTITY_DOC])

    @pytest.mark.asyncio
    async def test_raises_when_a_partition_has_two_documents(self) -> None:
        writer = FakeAsyncVaultWriter()
        dupe_user_doc = SeedDocument(
            seed_version="profile-v1",
            facts=[],
            body="a second, ambiguous user document",
            source_path=Path("mist-memory/seed/user2.md"),
            partition=ENTITY_LABEL,
        )

        with pytest.raises(SeedSourceError, match=ENTITY_LABEL):
            await admin.bootstrap_vault_from_seed(writer, [_IDENTITY_DOC, _USER_DOC, dupe_user_doc])
