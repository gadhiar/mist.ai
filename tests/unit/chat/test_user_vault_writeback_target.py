"""Tests for the C-pattern user-vault writeback TARGET.

`ConversationHandler._maybe_refresh_user_vault` renders a graph-derived
user snapshot after extraction and must persist it to a SEPARATE machine-
owned derived file (`users/<user_id>-graph-snapshot.md`) via
`VaultWriter.upsert_user_snapshot`. It must NOT call `upsert_user`, which
would clobber the hand-curated `users/<user_id>.md` (ADR-010: the curated
profile is user-authoritative).

The snapshot helpers (`extraction_touched_user_scope`, `query_user_snapshot`,
`render_user_snapshot_body`) are imported LAZILY inside the method body
(`from backend.vault.user_snapshot import ...` at call time), so they are not
bound to the `backend.chat.conversation_handler` module namespace. The correct
monkeypatch target is therefore `backend.vault.user_snapshot.<name>`.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from backend.chat.conversation_handler import ConversationHandler


class _FakeVaultWriter:
    """Explicit writer fake: records the two writeback calls the C-pattern
    can make. Defined inline (not promoted to tests/mocks/) per the scoped
    review fix. Unlike MagicMock, the keyword-only signatures here will fail
    the test if upsert_user_snapshot's interface drifts.
    """

    def __init__(self) -> None:
        self.upsert_user_snapshot_calls: list[tuple[str, str]] = []
        self.upsert_user_calls: list[tuple[str, str]] = []

    async def upsert_user_snapshot(self, *, user_id: str, body_markdown: str) -> str:
        self.upsert_user_snapshot_calls.append((user_id, body_markdown))
        return f"users/{user_id}-graph-snapshot.md"

    async def upsert_user(self, *, user_id: str, body_markdown: str) -> str:
        self.upsert_user_calls.append((user_id, body_markdown))
        return f"users/{user_id}.md"


def test_refresh_writes_snapshot_not_user_md(monkeypatch):
    # Arrange: construct a ConversationHandler without running __init__ and
    # wire only the attributes _maybe_refresh_user_vault reads.
    handler = ConversationHandler.__new__(ConversationHandler)
    fake_writer = _FakeVaultWriter()
    handler._vault_writer = fake_writer
    handler.graph_store = MagicMock()
    handler._user_id_for_vault = lambda: "user"

    extraction_result = MagicMock(
        validated_entities=[MagicMock()],
        validated_relationships=[],
    )

    # The helpers are imported lazily from backend.vault.user_snapshot inside
    # the method, so patch at that module (not on conversation_handler).
    monkeypatch.setattr(
        "backend.vault.user_snapshot.extraction_touched_user_scope",
        lambda *a, **k: True,
    )
    monkeypatch.setattr(
        "backend.vault.user_snapshot.query_user_snapshot",
        AsyncMock(return_value=MagicMock(edges_by_type={})),
    )
    monkeypatch.setattr(
        "backend.vault.user_snapshot.render_user_snapshot_body",
        lambda snapshot: "BODY",
    )

    # The method also builds a GraphExecutor(self.graph_store.connection);
    # stub it so no real Neo4j wiring is touched.
    monkeypatch.setattr(
        "backend.knowledge.storage.graph_executor.GraphExecutor",
        lambda *a, **k: MagicMock(),
    )

    # Act
    asyncio.run(handler._maybe_refresh_user_vault(extraction_result))

    # Assert: the snapshot writeback fired exactly once with the real user_id
    # and the rendered body; the curated user.md writer was never called.
    assert fake_writer.upsert_user_snapshot_calls == [("user", "BODY")]
    assert fake_writer.upsert_user_calls == []
